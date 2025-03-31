# Contents of package:
# Classes and functions related to ....
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import copy
import numpy as np
from .base import Parameters
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

@dataclass
class SCMs:
    Ryy: np.ndarray = None  # centralized SCM of the microphone signals
    Rss: np.ndarray = None  # centralized SCM of the desired source signals
    Rgkq: list = None  # SCM of the common sources between k and q
    Rykyqb: list = None  # SCM between zq and yk
    Rykykmq: list = None  # SCM of yk without the common sources between k and q
    Rykyk: list = None  # SCM of microphone signals at node k
    Rsksk: list = None  # SCM of the desired sources at node k
    Rnknk: list = None  # SCM of the noise sources at node k
    Rsslat: np.ndarray = None  # SCM of the latent desired sources
    Rnnlat: np.ndarray = None  # SCM of the latent noise sources
    Rvv: np.ndarray = None  # SCM of the self-noise

@dataclass
class Run:
    cfg: Parameters
    scms: SCMs = SCMs()

    def __post_init__(self):
        for field_name in SCMs.__dataclass_fields__:
            if getattr(self.scms, field_name) is None:
                if SCMs.__annotations__[field_name] != list:
                    continue
                setattr(
                    self.scms,
                    field_name,
                    [
                        [None for _ in range(self.cfg.K)]
                        for _ in range(self.cfg.K)
                    ]
                )
        pass

    def launch(self):
        c = self.cfg  # alias for convenience
        
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

        def _get_obs_matrices():
            """Generate random observability matrices."""
            oMatd = np.random.randint(0, 2, (c.Qd, c.K))
            oMatn = np.random.randint(0, 2, (c.Qn, c.K))
            # Ensure that at each source is observed by at least one node
            for i in range(c.Qd):
                if np.sum(oMatd[i, :]) == 0:
                    oMatd[i, np.random.randint(0, c.K)] = 1
            for i in range(c.Qn):
                if np.sum(oMatn[i, :]) == 0:
                    oMatn[i, np.random.randint(0, c.K)] = 1
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

        # Compute theoretical SCMs (stored in `self.scms`)
        self.compute_theoretical_scms(
            latd, latn, sn, Amat, Bmat, oMatd, oMatn, QkqMat
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
            'LCMP': copy.deepcopy(baseDict),
            'dMWF': copy.deepcopy(baseDict),
        }
        Pkq = [[{} for _ in range(c.K)] for _ in range(c.K)]
        for k in range(c.K):
            for q in range(c.K):
                if q == k:
                    continue
                # --- LCMV beamformer ---
                Qkq = QkqMat[k, q]  # number of common sources between k and q
                # Compute the yk vs. zq spatial covariance matrix (SCM)
                # and the unwanted signal SCM (oracle) <-- IMPORTANT ASSUMPTION for LCMV
                if c.scmEst == 'theoretical':
                    Rykyk = self.scms.Rykyk[k]
                    Rykyqb = self.scms.Rykyqb[k][q]
                    Rykykmq = self.scms.Rykykmq[k][q]
                elif c.scmEst == 'batch':
                    Rykyk = y[k] @ y[k].T / c.N
                    yqb = y[q][:Qkq, :]  # signal transmitted by q to k (Qkq x N)
                    Rykyqb = y[k] @ yqb.T / c.N
                    # Figure out which latent sources are common between k and q  # <-- IMPORTANT ASSUMPTION for dMWF
                    iComkqd = np.where(oMatd[:, k] & oMatd[:, q])[0]
                    iComkqn = np.where(oMatn[:, k] & oMatn[:, q])[0]
                    skq = Amat[k][:, iComkqd] @ latd[iComkqd, :]
                    nkq = Bmat[k][:, iComkqn] @ latn[iComkqn, :]
                    gkq = skq + nkq  # <-- also used for `Rgkq` below
                    iUncomkqd = np.delete(np.arange(c.Qd), iComkqd)
                    iUncomkqn = np.delete(np.arange(c.Qn), iComkqn)
                    skmq = Amat[k][:, iUncomkqd] @ latd[iUncomkqd, :]
                    nkmq = Bmat[k][:, iUncomkqn] @ latn[iUncomkqn, :]
                    gkmq = skmq + nkmq + sn[k]
                    assert np.allclose(gkq + gkmq, y[k])
                    Rykykmq = (y[k] - gkq) @ (y[k] - gkq).T / c.N
                Gam = np.linalg.inv(Rykykmq)
                Lam = np.linalg.inv(Rykyk)
                # LCMV beamformer
                Pkq[k][q]['LCMV'] = Gam @ Rykyqb @\
                    np.linalg.inv(Rykyqb.T @ Gam @ Rykyqb)  # (Mk x Qkq)
                Pkq[k][q]['LCMP'] = Lam @ Rykyqb #@\
                # Pkq[k][q]['LCMP'] = Lam[:, :Qkq] #@\
                    # np.linalg.inv(Rykyqb.T @ Lam @ Rykyqb)  # (Mk x Qkq)
                
                # --- dMWF original definition ---
                if c.scmEst == 'theoretical':
                    Rgkq = self.scms.Rgkq[k][q]
                elif c.scmEst == 'batch':
                    Rgkq = gkq @ gkq.T / c.N
                # Compute the MWF
                Ekloc = np.zeros((c.Mk, Qkq))
                Ekloc[:Qkq, :] = np.eye(Qkq)
                Pkq[k][q]['dMWF'] = np.linalg.inv(Rykyk) @\
                    Rgkq @ Ekloc # (Mk x Qkq)
                
                # Fused signals
                for BFtype in zkq.keys():
                    zkq[BFtype]['y'][k][q] = Pkq[k][q][BFtype].T @ y[k]
                    zkq[BFtype]['s'][k][q] = Pkq[k][q][BFtype].T @ s[k]
        
        # Compute target MWF at each node
        dhatk = dict([
            (BFtype, [None for _ in range(c.K)])
            for BFtype in zkq.keys()
        ])

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
                    Rty = Ck.T @ self.scms.Ryy @ Ck
                    Rts = Ck.T @ self.scms.Rss @ Ck
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
            Ryy = self.scms.Ryy
            Rss = self.scms.Rss
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
                Ryyloc = self.scms.Rykyk[k]
                Rssloc = self.scms.Rsksk[k]
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
        
    def compute_theoretical_scms(
            self,
            latd, latn, sn,
            Amat, Bmat,
            oMatd, oMatn,
            QkqMat
        ):
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
        Amat : list[np.ndarray]
            Steering matrices for the desired sources.
        Bmat : list[np.ndarray]
            Steering matrices for the noise sources.
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
        c = self.cfg  # alias for convenience
        s = self.scms  # alias for convenience
        # Latent SCMs
        # s.Rsslat = latd @ latd.T / c.N
        s.Rsslat = np.diag(np.diag(latd @ latd.T / c.N))
        # s.Rnnlat = latn @ latn.T / c.N
        s.Rnnlat = np.diag(np.diag(latn @ latn.T / c.N))
        snall = np.concatenate(sn, axis=0)
        # s.Rvv = snall @ snall.T / c.N
        s.Rvv = np.diag(np.diag(snall @ snall.T / c.N))
        Rvkvk = [s.Rvv[k * c.Mk:(k + 1) * c.Mk, k * c.Mk:(k + 1) * c.Mk] for k in range(c.K)]
        # Initialize lists
        Ak_q = [[None for _ in range(c.K)] for _ in range(c.K)]
        Bk_q = [[None for _ in range(c.K)] for _ in range(c.K)]
        Ak_mq = [[None for _ in range(c.K)] for _ in range(c.K)]
        Bk_mq = [[None for _ in range(c.K)] for _ in range(c.K)]
        Rsslatk_q = [[None for _ in range(c.K)] for _ in range(c.K)]
        Rnnlatk_q = [[None for _ in range(c.K)] for _ in range(c.K)]
        Rsslatk_mq = [[None for _ in range(c.K)] for _ in range(c.K)]
        Rnnlatk_mq = [[None for _ in range(c.K)] for _ in range(c.K)]
        for k in range(c.K):
            for q in range(c.K):
                if q == k:
                    continue
                # Find indices of global common sources between k and q
                iComkqd = np.where(oMatd[:, k] & oMatd[:, q])[0]
                iComkqn = np.where(oMatn[:, k] & oMatn[:, q])[0]
                # ...and of global sources _not_ common between k and q
                iUncomkqd = np.delete(np.arange(c.Qd), iComkqd)
                iUncomkqn = np.delete(np.arange(c.Qn), iComkqn)
                # Split the global-only steering matrices
                Ak_q[k][q] = Amat[k][:, iComkqd]
                Bk_q[k][q] = Bmat[k][:, iComkqn]
                Ak_mq[k][q] = Amat[k][:, iUncomkqd]
                Bk_mq[k][q] = Bmat[k][:, iUncomkqn]
                # Get corresponding latent signal SCMs
                Rsslatk_q[k][q] = np.diag(np.diag(s.Rsslat)[iComkqd])
                Rnnlatk_q[k][q] = np.diag(np.diag(s.Rnnlat)[iComkqn])
                Rsslatk_mq[k][q] = np.diag(np.diag(s.Rsslat)[iUncomkqd])
                Rnnlatk_mq[k][q] = np.diag(np.diag(s.Rnnlat)[iUncomkqn])

                # Compute kq-pair-specific local SCMs
                Rsksk_q = Ak_q[k][q] @ Rsslatk_q[k][q] @ Ak_q[k][q].T
                Rnknk_q = Bk_q[k][q] @ Rnnlatk_q[k][q] @ Bk_q[k][q].T
                Rsksk_mq = Ak_mq[k][q] @ Rsslatk_mq[k][q] @ Ak_mq[k][q].T
                Rnknk_mq = Bk_mq[k][q] @ Rnnlatk_mq[k][q] @ Bk_mq[k][q].T
                s.Rgkq[k][q] = Rsksk_q + Rnknk_q  # SCM of yk with just the contributions of common sources between k and q
                s.Rykykmq[k][q] = Rsksk_mq + Rnknk_mq + Rvkvk[k]  # SCM of yk without the common sources between k and q

                # Sanity checks
                assert np.allclose(  # Check consistency of sum of SCMs
                    s.Rgkq[k][q] + s.Rykykmq[k][q],
                    Amat[k] @ s.Rsslat @ Amat[k].T +\
                    Bmat[k] @ s.Rnnlat @ Bmat[k].T + Rvkvk[k]
                ), f"Error in SCM computation for k={k}, q={q}"

            # Compute usual local SCMs
            s.Rsksk[k] = Amat[k] @ s.Rsslat @ Amat[k].T
            s.Rnknk[k] = Bmat[k] @ s.Rnnlat @ Bmat[k].T
            s.Rykyk[k] = s.Rsksk[k] + s.Rnknk[k] + Rvkvk[k]
    
            # Check
            assert np.all([np.allclose(
                s.Rgkq[k][q] + s.Rykykmq[k][q],
                s.Rykyk[k]
            ) for q in range(c.K) if q != k])

        # Compute kq-pair-specific kq-SCMs
        for k in range(c.K):
            for q in range(c.K):
                if q == k:
                    continue
                Rsksq_com = Ak_q[k][q] @ Rsslatk_q[k][q] @ Ak_q[q][k].T
                Rnknsq_com = Bk_q[k][q] @ Rnnlatk_q[k][q] @ Bk_q[q][k].T
                s.Rykyqb[k][q] = (Rsksq_com + Rnknsq_com)[:, :QkqMat[k, q]]

        # Check symmetry of latent SCMs
        assert np.all([
            [
                np.allclose(Rsslatk_q[k][q], Rsslatk_q[q][k]) and\
                np.allclose(Rnnlatk_q[k][q], Rnnlatk_q[q][k]) and\
                np.allclose(Rsslatk_mq[k][q], Rsslatk_mq[q][k]) and\
                np.allclose(Rnnlatk_mq[k][q], Rnnlatk_mq[q][k])
                for k in range(c.K) if k != q
            ] for q in range(c.K)
        ]), "Error in SCM computation: not symmetric"
        
        # Tile matrices for centralized solution
        Ac = np.concatenate(Amat, axis=0)
        Bc = np.concatenate(Bmat, axis=0)
        # Centralized SCMs
        s.Ryy = Ac @ s.Rsslat @ Ac.T + Bc @ s.Rnnlat @ Bc.T + s.Rvv
        s.Rss = Ac @ s.Rsslat @ Ac.T