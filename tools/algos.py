# Contents of package:
# Classes and functions related to ....
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import copy
import numpy as np
import scipy.linalg as sla
from .base import Parameters
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Run:
    cfg: Parameters

    def launch(self):
        c = self.cfg  # alias for convenience

        # ---- Setup environment ----
        # Observability matrix: random matrix of 0's and 1's
        oMatd = np.random.randint(0, 2, (c.Qd, c.K))
        oMatn = np.random.randint(0, 2, (c.Qn, c.K))
        oMat = np.concatenate((oMatd, oMatn), axis=0)
        QkqMat = oMat.T @ oMat  # number of common sources between nodes

        # Centralized steering matrices
        Amat = [np.zeros((c.Mk, c.Qd)) for _ in range(c.K)]
        Bmat = [np.zeros((c.Mk, c.Qn)) for _ in range(c.K)]
        for k in range(c.K):
            # Fill in depending on observability
            Amat[k][:, oMatd[:, k].astype(bool)] = np.random.randn(
                c.Mk, np.sum(oMatd[:, k])
            )
            Bmat[k][:, oMatn[:, k].astype(bool)] = np.random.randn(
                c.Mk, np.sum(oMatn[:, k])
            )

        # Latent source signals
        latd = np.random.randn(c.Qd, c.N)  # desired
        latn = np.random.randn(c.Qn, c.N)  # noise
        # Self-noise
        sn = [np.random.randn(c.Mk, c.N) for _ in range(c.K)]
        # Microphone signals
        s = [Amat[k] @ latd for k in range(c.K)]  # desired sources contribution
        n = [Bmat[k] @ latn for k in range(c.K)]  # noise sources contribution
        d = [s[k][:c.D, :] for k in range(c.K)]  # target signals
        y = [s[k] + n[k] + sn[k] for k in range(c.K)]  # microphone signals

        # Fusion matrices and fused signals via LCMV beamforming or from the
        # dMWF basic definition (local MWFs)
        baseDict = {
            'y': [[None for _ in range(c.K)] for _ in range(c.K)],
            's': [[None for _ in range(c.K)] for _ in range(c.K)],
        }
        zkq = {
            'LCMV': baseDict,
            'dMWF': copy.deepcopy(baseDict),
        }
        
        for k in range(c.K):
            for q in range(c.K):
                if q == k:
                    continue
                Pkq = {}

                # --- LCMV beamformer ---
                Qkq = QkqMat[k, q]  # number of common sources between k and q
                zq = y[q][:Qkq, :]  # signal transmitted by q to k (Qkq x N)
                # Compute the spatial covariance matrix (SCM) (batch)
                Ryz = y[k] @ zq.T / c.N
                # Compute the unwanted signal SCM (oracle)
                Ru = (y[k] @ y[k].T - y[q] @ y[q].T) / c.N
                Gam = np.linalg.inv(Ru)
                # LCMV beamformer
                Pkq['LCMV'] = Gam @ Ryz @ np.linalg.inv(Ryz.T @ Gam @ Ryz)  # (Mk x Qkq)
                
                # --- dMWF basic definition ---
                Rykyk = y[k] @ y[k].T / c.N
                # Figure out which latent sources are common between k and q
                idxCommon_kq_d = np.where(oMatd[:, k] & oMatd[:, q])[0]
                idxCommon_kq_n = np.where(oMatn[:, k] & oMatn[:, q])[0]
                # if len(np.concatenate((idxCommon_kq_d, idxCommon_kq_n))) == 0:
                #     print(f"No common sources between nodes {k} and {q}.")
                skq = Amat[k][:, idxCommon_kq_d] @ latd[idxCommon_kq_d, :]
                nkq = Bmat[k][:, idxCommon_kq_n] @ latn[idxCommon_kq_n, :]
                gkq = skq + nkq + sn[k]
                Rgkq = gkq @ gkq.T / c.N
                # Compute the MWF
                Ekloc = np.zeros((c.Mk, Qkq))
                Ekloc[:Qkq, :] = np.eye(Qkq)
                Pkq['dMWF'] = np.linalg.inv(Rykyk) @ Rgkq @ Ekloc # (Mk x Qkq)
                
                # Fused signals
                for BFtype in zkq.keys():
                    zkq[BFtype]['y'][k][q] = Pkq[BFtype].T @ y[k]
                    zkq[BFtype]['s'][k][q] = Pkq[BFtype].T @ s[k]
        
        # Compute target MWF at each node
        dhatk = dict([
            (BFtype, [None for _ in range(c.K)])
            for BFtype in zkq.keys()
        ])

        def build_tilde(yk, zkqCurr, kCurr):
            zz = np.concatenate(
                [zkqCurr[q][kCurr] for q in range(c.K) if q != kCurr],
                axis=0
            )
            return np.concatenate([yk, zz], axis=0)
        
        for k in range(c.K):
            for BFtype in zkq.keys():
                ty = build_tilde(y[k], zkq[BFtype]['y'], k)
                ts = build_tilde(s[k], zkq[BFtype]['s'], k)
                Rty = ty @ ty.T / c.N
                Rts = ts @ ts.T / c.N
                # Selection vector
                tE = np.zeros((Rty.shape[0], c.D))
                tE[:c.D, :] = np.eye(c.D)
                tWk = np.linalg.inv(Rty) @ Rts @ tE
                # Compute target signal estimate
                dhatk[BFtype][k] = tWk.T @ ty

        # Compute centralized and local MWFs
        yc = np.concatenate(y, axis=0)
        sc = np.concatenate(s, axis=0)
        Ryy = yc @ yc.T / c.N
        Rss = sc @ sc.T / c.N
        dhatk['Centralized'] = [None for _ in range(c.K)]
        dhatk['Local'] = [None for _ in range(c.K)]
        for k in range(c.K):
            Ek = np.zeros((Ryy.shape[0], c.D))
            Ek[k * c.Mk:k * c.Mk + c.D, :] = np.eye(c.D)
            Wk = np.linalg.inv(Ryy) @ Rss @ Ek
            dhatk['Centralized'][k] = Wk.T @ yc
            # Local MWF
            Rykyk = y[k] @ y[k].T / c.N
            Rsksk = s[k] @ s[k].T / c.N
            Ekk = np.zeros((Rykyk.shape[0], c.D))
            Ekk[:c.D, :] = np.eye(c.D)
            Wkloc = np.linalg.inv(Rykyk) @ Rsksk @ Ekk
            dhatk['Local'][k] = Wkloc.T @ y[k]

        # Compute MSE_d
        msed = dict([
            (BFtype, [
                np.mean(dh[k] - d[k]) ** 2
                for k in range(c.K)
            ]) for BFtype, dh in dhatk.items()
        ])
        # Print nice results
        # for BFtype, mse in msed.items():
        #     print(f"MSE_d ({BFtype}): {mse}")

        return msed
        