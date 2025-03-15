# Contents of package:
# Classes and functions related to ....
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import copy
import numpy as np
from .base import Parameters
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

        # Latent source signals
        latd = np.random.randn(c.Qd, c.N)  # desired
        latn = np.random.randn(c.Qn, c.N)  # noise
        # Self-noise
        sn = [np.random.randn(c.Mk, c.N) for _ in range(c.K)]

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

        # Compute theoretical SCMs
        Rgkq, Rykzq, Rykyk, Rykykmq, Rsslat, Rnnlat, Rvv, Ac, Bc =\
            self.compute_theoretical_scms(
            latd, latn, sn, oMatd, oMatn, QkqMat
        )

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
                # Compute the spatial covariance matrix (SCM)
                if c.scmEst == 'theoretical':
                    Ryz = Rykzq[k][q]
                elif c.scmEst == 'batch':
                    zq = y[q][:Qkq, :]  # signal transmitted by q to k (Qkq x N)
                    Ryz = y[k] @ zq.T / c.N
                # Compute the unwanted signal SCM (oracle) <-- IMPORTANT ASSUMPTION for LCMV
                if c.scmEst == 'theoretical':
                    Ru = Rykykmq[k][q]
                elif c.scmEst == 'batch':
                    Ru = (y[k] @ y[k].T - y[q] @ y[q].T) / c.N
                Gam = np.linalg.inv(Ru)
                # LCMV beamformer
                Pkq['LCMV'] = Gam @ Ryz @ np.linalg.inv(Ryz.T @ Gam @ Ryz)  # (Mk x Qkq)
                
                # --- dMWF basic definition ---
                if c.scmEst == 'theoretical':
                    RyyLoc = Rykyk[k]
                elif c.scmEst == 'batch':
                    RyyLoc = y[k] @ y[k].T / c.N
                # Figure out which latent sources are common between k and q  # <-- IMPORTANT ASSUMPTION for dMWF
                if c.scmEst == 'theoretical':
                    RgkqCurr = Rgkq[k][q]
                elif c.scmEst == 'batch':
                    idxCommon_kq_d = np.where(oMatd[:, k] & oMatd[:, q])[0]
                    idxCommon_kq_n = np.where(oMatn[:, k] & oMatn[:, q])[0]
                    skq = Amat[k][:, idxCommon_kq_d] @ latd[idxCommon_kq_d, :]
                    nkq = Bmat[k][:, idxCommon_kq_n] @ latn[idxCommon_kq_n, :]
                    gkq = skq + nkq + sn[k]
                    RgkqCurr = gkq @ gkq.T / c.N
                # Compute the MWF
                Ekloc = np.zeros((c.Mk, Qkq))
                Ekloc[:Qkq, :] = np.eye(Qkq)
                Pkq['dMWF'] = np.linalg.inv(RyyLoc) @ RgkqCurr @ Ekloc # (Mk x Qkq)
                
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
                if c.scmEst == 'theoretical':
                    Rty = None  # TODO: theoretical global MWF SCM, via Ck matrix and Pk matrices
                elif c.scmEst == 'batch':
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
        if c.scmEst == 'theoretical':
            Ryy = Ac @ Rsslat @ Ac.T + Bc @ Rnnlat @ Bc.T + Rvv
            Rss = Ac @ Rsslat @ Ac.T
        elif c.scmEst == 'batch':
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
            if c.scmEst == 'theoretical':
                Ryyloc = Rykyk[k]
                Rssloc = Amat[k] @ Rsslat @ Amat[k].T
            elif c.scmEst == 'batch':
                Ryyloc = y[k] @ y[k].T / c.N
                Rssloc = s[k] @ s[k].T / c.N
            Ekk = np.zeros((Ryyloc.shape[0], c.D))
            Ekk[:c.D, :] = np.eye(c.D)
            Wkloc = np.linalg.inv(Ryyloc) @ Rssloc @ Ekk
            dhatk['Local'][k] = Wkloc.T @ y[k]

        # Compute unprocessed MSEd
        dhatk['Unprocessed'] = [y[k][:c.D, :] for k in range(c.K)]  # unprocessed signal

        # Compute MSE_d
        msed = dict([
            (BFtype, [
                np.mean(dh[k] - d[k]) ** 2
                for k in range(c.K)
            ]) for BFtype, dh in dhatk.items()
        ])

        return msed
        
    def compute_theoretical_scms(self, latd, latn, sn, oMatd, oMatn, QkqMat):
        c = self.cfg 
        # Latent SCMs
        Rsslat = np.diag(np.mean(latd, axis=1) ** 2)
        Rnnlat = np.diag(np.mean(latn, axis=1) ** 2)
        Rvv = np.diag(np.mean(np.concatenate(sn, axis=0), axis=1) ** 2)
        Rvkvk = [Rvv[k * c.Mk:(k + 1) * c.Mk,k * c.Mk:(k + 1) * c.Mk] for k in range(c.K)]
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
        # Rearrange the matrices colums to have the sources that are only
        # observed by one node at the end
        allIdx = np.arange(c.Qd)
        idxSingle_d = np.where(np.sum(oMatd, axis=1) == 1)[0]  # desired sources observed by only one node
        idxSingle_n = np.where(np.sum(oMatn, axis=1) == 1)[0]  # noise sources observed by only one node
        for k in range(c.K):
            Amat[k] = np.concatenate((
                Amat[k][:, np.delete(allIdx, idxSingle_d)],
                Amat[k][:, idxSingle_d]
            ), axis=1)
            Bmat[k] = np.concatenate((
                Bmat[k][:, np.delete(allIdx, idxSingle_n)],
                Bmat[k][:, idxSingle_n]
            ), axis=1)
        # Tile matrices for centralized solution
        Ac = np.concatenate(Amat, axis=0)
        Bc = np.concatenate(Bmat, axis=0)
        # Split the matrices into local and global contribution parts
        cA = [Amat[k][:, :len(idxSingle_d)] for k in range(c.K)]
        uA = [Amat[k][:, len(idxSingle_d):] for k in range(c.K)]
        cB = [Bmat[k][:, :len(idxSingle_n)] for k in range(c.K)]
        uB = [Bmat[k][:, len(idxSingle_n):] for k in range(c.K)]
        cAkq = [[None for _ in range(c.K)] for _ in range(c.K)]
        uAkq = [[None for _ in range(c.K)] for _ in range(c.K)]
        cBkq = [[None for _ in range(c.K)] for _ in range(c.K)]
        uBkq = [[None for _ in range(c.K)] for _ in range(c.K)]
        cAkmq = [[None for _ in range(c.K)] for _ in range(c.K)]
        uAkmq = [[None for _ in range(c.K)] for _ in range(c.K)]
        cBkmq = [[None for _ in range(c.K)] for _ in range(c.K)]
        uBkmq = [[None for _ in range(c.K)] for _ in range(c.K)]
        Rsslatkq = [[None for _ in range(c.K)] for _ in range(c.K)]
        Rnnlatkq = [[None for _ in range(c.K)] for _ in range(c.K)]
        Rsslatkmq = [[None for _ in range(c.K)] for _ in range(c.K)]
        Rnnlatkmq = [[None for _ in range(c.K)] for _ in range(c.K)]
        for k in range(c.K):
            for q in range(c.K):
                if q == k:
                    continue
                # Find indices of common sources between k and q
                idxCommon_kq_d = np.where(oMatd[:, k] & oMatd[:, q])[0]
                idxCommon_kq_n = np.where(oMatn[:, k] & oMatn[:, q])[0]
                idxUncommon_kq_d = np.delete(allIdx, idxCommon_kq_d)
                idxUncommon_kq_n = np.delete(allIdx, idxCommon_kq_n)
                # Split the matrices
                cAkq[k][q] = cA[k][:, idxCommon_kq_d]
                uAkq[k][q] = uA[k][:, idxCommon_kq_d]
                cBkq[k][q] = cB[k][:, idxCommon_kq_n]
                uBkq[k][q] = uB[k][:, idxCommon_kq_n]
                cAkmq[k][q] = cA[q][:, idxUncommon_kq_d]
                uAkmq[k][q] = uA[q][:, idxUncommon_kq_d]
                cBkmq[k][q] = cB[q][:, idxUncommon_kq_n]
                uBkmq[k][q] = uB[q][:, idxUncommon_kq_n]
                # Compute latent signal SCMs
                Rsslatkq[k][q] = np.diag(np.diag(Rsslat)[idxCommon_kq_d])
                Rnnlatkq[k][q] = np.diag(np.diag(Rnnlat)[idxCommon_kq_n])
                Rsslatkmq[k][q] = np.diag(np.diag(Rsslat)[idxUncommon_kq_d])
                Rnnlatkmq[k][q] = np.diag(np.diag(Rnnlat)[idxUncommon_kq_n])
        
        Rgkq = [
            [
                cAkq[k][q] @ Rsslatkq[k][q] @ cAkq[k][q].T +\
                    cBkq[k][q] @ Rnnlatkq[k][q] @ cBkq[k][q].T
                if q != k else None
                for q in range(c.K)
            ]
            for k in range(c.K)
        ]
        Rykzq = [
            [
                Rgkq[k][q][:, :QkqMat[k, q]]
                if q != k else None
                for q in range(c.K)
            ]
            for k in range(c.K)
        ]
        Rykyk = [
            Amat[k] @ Rsslat @ Amat[k].T + Bmat[k] @ Rnnlat @ Bmat[k].T + Rvkvk[k]
            for k in range(c.K)
        ]
        Rykykmq = [
            [
                cAkmq[k][q] @ Rsslatkmq[k][q] @ cAkmq[k][q].T +\
                    cBkmq[k][q] @ Rnnlatkmq[k][q] @ cBkmq[k][q].T + Rvkvk[k]
                if q != k else None
                for q in range(c.K)
            ]
            for k in range(c.K)
        ]

        return Rgkq, Rykzq, Rykyk, Rykykmq, Rsslat, Rnnlat, Rvv, Ac, Bc