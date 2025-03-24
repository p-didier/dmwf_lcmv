# Contents of package:
# Classes and functions related to ....
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import copy
import numpy as np
from .base import Parameters
from dataclasses import dataclass

@dataclass
class SCMs:
    Ryy: np.ndarray = None  # centralized SCM of the microphone signals
    Rss: np.ndarray = None  # centralized SCM of the desired source signals
    Rgkq: list = None  # SCM of the common sources between k and q
    Rykzq: list = None  # SCM between zq and yk
    Rykykmq: list = None  # SCM of yk without the common sources between k and q
    Rykyk: list = None  # SCM of microphone signals at node k
    Rsksk: list = None  # SCM of the desired sources at node k
    Rsslat: np.ndarray = None  # SCM of the latent desired sources
    Rnnlat: np.ndarray = None  # SCM of the latent noise sources
    Rvv: np.ndarray = None  # SCM of the self-noise

@dataclass
class Run:
    cfg: Parameters

    def launch(self):
        c = self.cfg  # alias for convenience

        def _get_obs_matrices():
            """Generate random observability matrices."""
            oMatd = np.random.randint(0, 2, (c.Qd, c.K))
            oMatn = np.random.randint(0, 2, (c.Qn, c.K))
            # Ensure that at least one source is observed by each node
            oMatd[np.random.randint(0, c.Qd), :] = 1
            oMatn[np.random.randint(0, c.Qn), :] = 1
            # Ensure that at least one source is observed by only one node
            idxSingle_d = np.random.randint(0, c.Qd)
            idxSingle_n = np.random.randint(0, c.Qn)
            oMatd[idxSingle_d, :] = 0
            oMatd[idxSingle_d, np.random.randint(0, c.K)] = 1
            oMatn[idxSingle_n, :] = 0
            oMatn[idxSingle_n, np.random.randint(0, c.K)] = 1
            return oMatd, oMatn

        # ---- Setup environment ----
        # Observability matrix: random matrix of 0's and 1's
        oMatd, oMatn = _get_obs_matrices()
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
        scms = self.compute_theoretical_scms(
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
        Pkq = [[{} for _ in range(c.K)] for _ in range(c.K)]
        for k in range(c.K):
            for q in range(c.K):
                if q == k:
                    continue
                # --- LCMV beamformer ---
                Qkq = QkqMat[k, q]  # number of common sources between k and q
                # Compute the spatial covariance matrix (SCM)
                if c.scmEst == 'theoretical':
                    Ryz = scms.Rykzq[k][q]
                elif c.scmEst == 'batch':
                    zq = y[q][:Qkq, :]  # signal transmitted by q to k (Qkq x N)
                    Ryz = y[k] @ zq.T / c.N
                pass
                # Compute the unwanted signal SCM (oracle) <-- IMPORTANT ASSUMPTION for LCMV
                if c.scmEst == 'theoretical':
                    Ru = scms.Rykykmq[k][q]
                elif c.scmEst == 'batch':
                    Ru = (y[k] @ y[k].T - y[q] @ y[q].T) / c.N
                Gam = np.linalg.inv(Ru)
                # LCMV beamformer
                Pkq[k][q]['LCMV'] = Gam @ Ryz @ np.linalg.inv(Ryz.T @ Gam @ Ryz)  # (Mk x Qkq)
                
                # --- dMWF basic definition ---
                if c.scmEst == 'theoretical':
                    RyyLoc = scms.Rykyk[k]
                elif c.scmEst == 'batch':
                    RyyLoc = y[k] @ y[k].T / c.N
                # Figure out which latent sources are common between k and q  # <-- IMPORTANT ASSUMPTION for dMWF
                if c.scmEst == 'theoretical':
                    RgkqCurr = scms.Rgkq[k][q]
                elif c.scmEst == 'batch':
                    idxCommon_kq_d = np.where(oMatd[:, k] & oMatd[:, q])[0]
                    idxCommon_kq_n = np.where(oMatn[:, k] & oMatn[:, q])[0]
                    skq = Amat[k][:, idxCommon_kq_d] @ latd[idxCommon_kq_d, :]
                    nkq = Bmat[k][:, idxCommon_kq_n] @ latn[idxCommon_kq_n, :]
                    gkq = skq + nkq# + sn[k]
                    RgkqCurr = gkq @ gkq.T / c.N
                # Compute the MWF
                Ekloc = np.zeros((c.Mk, Qkq))
                Ekloc[:Qkq, :] = np.eye(Qkq)
                Pkq[k][q]['dMWF'] = np.linalg.inv(RyyLoc) @ RgkqCurr @ Ekloc # (Mk x Qkq)
                
                # Fused signals
                for BFtype in zkq.keys():
                    zkq[BFtype]['y'][k][q] = Pkq[k][q][BFtype].T @ y[k]
                    zkq[BFtype]['s'][k][q] = Pkq[k][q][BFtype].T @ s[k]
        
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

        def build_Ck(k: int, Pkq: list[np.ndarray]):
            """
            Build the C_k matrix for the theoretical dMWF SCMs, such that
            \\tilde{y}_k = C_k^H y, where y = [y_1^T, ..., y_K^T]^T.
            C_k is thus a (M x (Mk + Qcomb)) matrix, where Qcomb is the
            combined number of fused signals channels received by node k. 
            
            Parameters
            ----------
            k : int
                Current node index.
            Pkq : list[np.ndarray]
                List of Pkq matrices for the dMWF at node k.
            """
            # Dimensions of zq signals
            zqDims = [p.shape[1] if p is not None else 0 for p in Pkq]
            Ck = np.zeros((c.M, c.Mk + np.sum(zqDims)))
            # Fill in the Pkq matrices
            Ck[k * c.Mk:(k + 1) * c.Mk, :c.Mk] = np.eye(c.Mk)
            for q in range(c.K):
                if q == k:
                    continue
                Ck[
                    q * c.Mk:(q + 1) * c.Mk,
                    c.Mk + int(np.sum(zqDims[:q])):\
                        c.Mk + int(np.sum(zqDims[:q + 1]))
                ] = Pkq[q]
            return Ck
        
        for k in range(c.K):
            for BFtype in zkq.keys():
                ty = build_tilde(y[k], zkq[BFtype]['y'], k)
                ts = build_tilde(s[k], zkq[BFtype]['s'], k)
                if c.scmEst == 'theoretical':
                    Ck = build_Ck(k, [
                        Pkq[k][q][BFtype]
                        if q != k else None
                        for q in range(c.K)
                    ])
                    Rty = Ck.T @ scms.Ryy @ Ck
                    Rts = Ck.T @ scms.Rss @ Ck
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
            Ryy = scms.Ryy
            Rss = scms.Rss
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
                Ryyloc = scms.Rykyk[k]
                Rssloc = scms.Rsksk[k]
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
        """
        Compute the theoretical SCMs for the given environment.

        Parameters
        ----------
        latd : np.ndarray
            Desired latent source signals.
        latn : np.ndarray
            Noise latent source signals.
        sn : list[np.ndarray]
            Self-noise signals.
        oMatd : np.ndarray (Qd x K)
            Observability matrix for the desired sources.
        oMatn : np.ndarray (Qn x K)
            Observability matrix for the noise sources.
        QkqMat : np.ndarray (K x K)
            Matrix of the number of common sources between nodes.

        Returns
        -------
        scms : SCMs object
            Theoretical SCMs object.
        """
        c = self.cfg 
        # Initialize SCMs
        scms = SCMs()
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
        # Rearrange the steering matrices colums and latent signal entries to
        # have the sources that are only observed by one node at the end.
        allDesIdx = np.arange(c.Qd)
        iSingd = np.where(np.sum(oMatd, axis=1) == 1)[0]  # indices of desired sources observed by only one node
        iSingn = np.where(np.sum(oMatn, axis=1) == 1)[0]  # indices of noise sources observed by only one node
        iMultd = np.delete(allDesIdx, iSingd)
        iMultn = np.delete(allDesIdx, iSingn)
        # Rearrange the steering matrices and latent signals
        latd = np.concatenate((latd[iMultd, :], latd[iSingd, :]), axis=0)
        latn = np.concatenate((latn[iMultn, :], latn[iSingn, :]), axis=0)
        for k in range(c.K):
            Amat[k] = np.concatenate((
                Amat[k][:, iMultd],
                Amat[k][:, iSingd]
            ), axis=1)
            Bmat[k] = np.concatenate((
                Bmat[k][:, iMultn],
                Bmat[k][:, iSingn]
            ), axis=1)
        # Latent SCMs
        scms.Rsslat = np.diag(np.mean(latd, axis=1) ** 2)
        scms.Rnnlat = np.diag(np.mean(latn, axis=1) ** 2)
        scms.Rvv = np.diag(np.mean(np.concatenate(sn, axis=0), axis=1) ** 2)
        Rvkvk = [scms.Rvv[k * c.Mk:(k + 1) * c.Mk,k * c.Mk:(k + 1) * c.Mk] for k in range(c.K)]
        # Tile matrices for centralized solution
        Ac = np.concatenate(Amat, axis=0)
        Bc = np.concatenate(Bmat, axis=0)
        # Extract global contribution part
        cAk = [Amat[k][:, :-len(iSingd)] for k in range(c.K)]
        cBk = [Bmat[k][:, :-len(iSingn)] for k in range(c.K)]
        cAkq = [[None for _ in range(c.K)] for _ in range(c.K)]
        cBkq = [[None for _ in range(c.K)] for _ in range(c.K)]
        cAkmq = [[None for _ in range(c.K)] for _ in range(c.K)]
        cBkmq = [[None for _ in range(c.K)] for _ in range(c.K)]
        Rsslatkq = [[None for _ in range(c.K)] for _ in range(c.K)]
        Rnnlatkq = [[None for _ in range(c.K)] for _ in range(c.K)]
        Rsslatkmq = [[None for _ in range(c.K)] for _ in range(c.K)]
        Rnnlatkmq = [[None for _ in range(c.K)] for _ in range(c.K)]
        for k in range(c.K):
            for q in range(c.K):
                if q == k:
                    continue
                # Find indices of global common sources between k and q
                iComkqd = np.where(oMatd[iMultd, k] & oMatd[iMultd, q])[0]
                iComkqn = np.where(oMatn[iMultn, k] & oMatn[iMultn, q])[0]
                # ...and of global sources _not_ common between k and q
                iUncomkqd = np.delete(np.arange(len(iMultd)), iComkqd)
                iUncomkqn = np.delete(np.arange(len(iMultn)), iComkqn)
                # Split the global-only steering matrices
                cAkq[k][q] = cAk[k][:, iComkqd]
                cBkq[k][q] = cBk[k][:, iComkqn]
                cAkmq[k][q] = cAk[k][:, iUncomkqd]
                cBkmq[k][q] = cBk[k][:, iUncomkqn]
                # Compute latent signal SCMs
                Rsslatkq[k][q] = np.diag(scms.Rsslat[iMultd, iMultd][iComkqd])
                Rnnlatkq[k][q] = np.diag(scms.Rnnlat[iMultn, iMultn][iComkqn])
                Rsslatkmq[k][q] = np.diag(scms.Rsslat[iMultd, iMultd][iUncomkqd])
                Rnnlatkmq[k][q] = np.diag(scms.Rnnlat[iMultn, iMultn][iUncomkqn])
        
        scms.Rgkq = [
            [
                cAkq[k][q] @ Rsslatkq[k][q] @ cAkq[k][q].T +\
                    cBkq[k][q] @ Rnnlatkq[k][q] @ cBkq[k][q].T
                if q != k else None
                for q in range(c.K)
            ]
            for k in range(c.K)
        ]
        scms.Rykzq = [
            [
                scms.Rgkq[k][q][:, :QkqMat[k, q]]
                if q != k else None
                for q in range(c.K)
            ]
            for k in range(c.K)
        ]
        scms.Rsksk = [
            Amat[k] @ scms.Rsslat @ Amat[k].T
            for k in range(c.K)
        ]
        scms.Rykyk = [
            scms.Rsksk[k] + Bmat[k] @ scms.Rnnlat @ Bmat[k].T + Rvkvk[k]
            for k in range(c.K)
        ]
        scms.Rykykmq = [
            [
                cAkmq[k][q] @ Rsslatkmq[k][q] @ cAkmq[k][q].T +\
                    cBkmq[k][q] @ Rnnlatkmq[k][q] @ cBkmq[k][q].T + Rvkvk[k]
                if q != k else None
                for q in range(c.K)
            ]
            for k in range(c.K)
        ]

        # Centralized SCMs
        scms.Ryy = Ac @ scms.Rsslat @ Ac.T + Bc @ scms.Rnnlat @ Bc.T + scms.Rvv
        scms.Rss = Ac @ scms.Rsslat @ Ac.T

        return scms