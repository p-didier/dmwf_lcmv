# Contents of package:
# Classes and functions related to ....
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import copy
import numpy as np
from .base import Parameters
import matplotlib.pyplot as plt
from pyinstrument import Profiler
from dataclasses import dataclass, field

@dataclass
class SCM:
    val: np.ndarray = None
    beta: float = 0.995  # Exponential averaging factor
    def update(self, yyH: np.ndarray):
        """Update the spatial covariance matrix using exponential averaging."""
        if self.val is None:
            # SCM not yet initialized. Do it now
            self.val = np.zeros_like(yyH)
        self.val = self.beta * self.val + (1 - self.beta) * yyH

@dataclass
class TheoreticalSCMs:
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
    scms: TheoreticalSCMs = TheoreticalSCMs()

    def __post_init__(self):
        for field_name in TheoreticalSCMs.__dataclass_fields__:
            if getattr(self.scms, field_name) is None:
                if TheoreticalSCMs.__annotations__[field_name] != list:
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

    def _get_steering_matrices(self, oMatd, oMatn):
        """
        Compute the steering matrices for the desired and noise sources.

        Parameters
        ----------
        oMatd : np.ndarray (Qd x K)
            Observability matrix for the desired sources.
        oMatn : np.ndarray (Qn x K)
            Observability matrix for the noise sources.

        Returns
        -------
        Amat : list[np.ndarray]
            Steering matrices for the desired sources.
        Bmat : list[np.ndarray]
            Steering matrices for the noise sources.
        """
        c = self.cfg  # alias for convenience
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
        return Amat, Bmat

    def _get_latent_sigs(self, nSamples: int, Amat, Bmat):
        """
        Compute latent source signals and corresponding steering
        matrices.
        
        Parameters
        ----------
        nSamples : int
            Number of samples to generate per channel.
        Amat : list[np.ndarray]
            Steering matrices for the desired sources.
        Bmat : list[np.ndarray]
            Steering matrices for the noise sources.
        
        Returns
        -------
        y : list[np.ndarray]
            Microphone signals per node.
        s : list[np.ndarray]
            Desired source signals per node.
        d : list[np.ndarray]
            Target signals per node.
        n : list[np.ndarray]
            Noise signals per node.
        latd : np.ndarray
            Desired latent source signals.
        latn : np.ndarray
            Noise latent source signals.
        sn : list[np.ndarray]
            Self-noise signals per node.
        """
        c = self.cfg  # alias for convenience
        # Latent source signals
        latd = np.random.randn(c.Qd, nSamples)  # desired
        latn = np.random.randn(c.Qn, nSamples)  # noise
        # Self-noise
        sn = [np.random.randn(c.Mk, nSamples) for _ in range(c.K)]
        # Microphone signals
        s = [Amat[k] @ latd for k in range(c.K)]  # desired sources contribution
        n = [Bmat[k] @ latn for k in range(c.K)]  # noise sources contribution
        d = [s[k][:c.D, :] for k in range(c.K)]  # target signals
        y = [s[k] + n[k] + sn[k] for k in range(c.K)]  # microphone signals
        return y, s, d, n, latd, latn, sn
        
    def build_tilde(self, yk, zkqCurr, kCurr):
        """Build \\tilde{y}_k (DANSE or dMWF)."""
        zmk = np.concatenate(
            [
                zkqCurr[q][kCurr]
                for q in range(self.cfg.K)
                if q != kCurr
            ],
            axis=0
        )
        return np.concatenate([yk, zmk], axis=0)

    def build_Ck(self, k: int, Pkq: list[np.ndarray]):
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
        c = self.cfg  # alias for convenience
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

    def obs_matrices(self):
        """Generate random observability matrices."""
        c = self.cfg  # alias for convenience
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
        
    def launch(self):
        if self.cfg.scmEst in ['theoretical', 'batch']:
            return self.launch_batch_type()
        else:
            return self.launch_online_type()

    def launch_batch_type(self):
        """
        Launch the batch type simulation (theoretical or batch SCM estimation).
        """
        c = self.cfg  # alias for convenience

        oMatd, oMatn = self.obs_matrices()
        oMat = np.concatenate((oMatd, oMatn), axis=0)
        QkqMat = oMat.T @ oMat  # number of common sources between nodes

        # Compute signals at once
        Amat, Bmat = self._get_steering_matrices(oMatd, oMatn)
        y, s, d, n, latd, latn, sn =\
            self._get_latent_sigs(
                nSamples=c.Nbatch,
                Amat=Amat,
                Bmat=Bmat
            )

        # Fusion matrices and fused signals via LCMV beamforming or from the
        # dMWF basic definition (local MWFs)
        baseDict = {
            'y': [[None for _ in range(c.K)] for _ in range(c.K)],
            's': [[None for _ in range(c.K)] for _ in range(c.K)],
        }
        zkq = {
            'LCMV': baseDict,
            'Simple': copy.deepcopy(baseDict),
            'dMWF': copy.deepcopy(baseDict),
        }
        Pkq = [[{} for _ in range(c.K)] for _ in range(c.K)]
        for k in range(c.K):
            for q in range(c.K):
                if q == k:
                    continue
                # --- LCMV beamformer ---
                Qkq = QkqMat[k, q]  # number of common sources between k and q
                if c.scmEst == 'theoretical':
                    Rykyk = self.scms.Rykyk[k]
                    Rykyqb = self.scms.Rykyqb[k][q]
                    Rykykmq = self.scms.Rykykmq[k][q]
                elif c.scmEst == 'batch':
                    Rykyk = y[k] @ y[k].T / c.Nbatch
                    yqb = y[q][:Qkq, :]  # signal transmitted by q to k (Qkq x N)
                    Rykyqb = y[k] @ yqb.T / c.Nbatch
                    # Figure out which latent sources are common between k and q  # <-- IMPORTANT ASSUMPTION for dMWF
                    gkq = get_gkq(k, q, Amat, Bmat, latd, latn, oMatd, oMatn)
                    Rykykmq = (y[k] - gkq) @ (y[k] - gkq).T / c.Nbatch
                Gam = np.linalg.inv(Rykykmq)
                Lam = np.linalg.inv(Rykyk)
                # LCMV beamformer
                Pkq[k][q]['LCMV'] = Gam @ Rykyqb @\
                    np.linalg.inv(Rykyqb.T @ Gam @ Rykyqb)  # (Mk x Qkq)
                Pkq[k][q]['Simple'] = Lam @ Rykyqb

                # --- dMWF original definition ---
                if c.scmEst == 'theoretical':
                    Rgkq = self.scms.Rgkq[k][q]
                elif c.scmEst == 'batch':
                    Rgkq = gkq @ gkq.T / c.Nbatch
                Ekloc = np.zeros((c.Mk, Qkq))
                Ekloc[:Qkq, :] = np.eye(Qkq)
                Pkq[k][q]['dMWF'] = np.linalg.inv(Rykyk) @\
                    Rgkq @ Ekloc # (Mk x Qkq)
                
                pass
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
                ty = self.build_tilde(y[k], zkq[BFtype]['y'], k)
                ts = self.build_tilde(s[k], zkq[BFtype]['s'], k)
                if c.scmEst == 'theoretical':
                    Ck = self.build_Ck(k, [
                        Pkq[k][q][BFtype]
                        if q != k else None
                        for q in range(c.K)
                    ])
                    Rty = Ck.T @ self.scms.Ryy @ Ck
                    Rts = Ck.T @ self.scms.Rss @ Ck
                elif c.scmEst == 'batch':
                    Rty = ty @ ty.T / c.Nbatch
                    Rts = ts @ ts.T / c.Nbatch
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
            Ryy = yc @ yc.T / c.Nbatch
            Rss = sc @ sc.T / c.Nbatch
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
                Ryyloc = y[k] @ y[k].T / c.Nbatch
                Rssloc = s[k] @ s[k].T / c.Nbatch
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
    
    def launch_online_type(self):
        """
        Launch the online type simulation (online SCM estimation).
        """
        c = self.cfg  # alias for convenience

        # Set up environment
        oMatd, oMatn = self.obs_matrices()
        oMat = np.concatenate((oMatd, oMatn), axis=0)
        QkqMat = oMat.T @ oMat  # number of common sources between nodes
        Amat, Bmat = self._get_steering_matrices(oMatd, oMatn)

        # Fusion matrices and fused signals via LCMV beamforming or from the
        # dMWF basic definition (local MWFs)
        baseDict = {
            'y': [
                [np.random.randn(QkqMat[k, q], c.Nonline) for q in range(c.K)]
                for k in range(c.K)
            ],
            's': [
                [np.random.randn(QkqMat[k, q], c.Nonline) for q in range(c.K)]
                for k in range(c.K)
            ],
        }
        zkq = {
            'LCMV': baseDict,
            'LCMP': copy.deepcopy(baseDict),
            'Simple': copy.deepcopy(baseDict),
            'dMWF': copy.deepcopy(baseDict),
            'DANSE': copy.deepcopy(baseDict),
        }
        Pkq = [[dict([
            (BFtype, np.random.randn(c.Mk, QkqMat[k, q]))
            if BFtype != 'DANSE' else
            (BFtype, np.random.randn(c.Mk, c.Qd))
            for BFtype in zkq.keys()
        ]) for q in range(c.K)] for k in range(c.K)]
        tWk = [  # DANSE filter \tilde{W}_k
            np.random.randn(c.Mk + c.Qd * (c.K - 1), c.Qd)
            for _ in range(c.K)
        ]
        # SCMs
        Ryy = SCM(beta=c.beta)
        Rss = SCM(beta=c.beta)
        baseList = [SCM(beta=c.beta) for _ in range(c.K)]
        Rty = dict([(BFtype, copy.deepcopy(baseList)) for BFtype in zkq.keys()])
        Rts = copy.deepcopy(Rty)
        Rykyk = copy.deepcopy(baseList)
        Rsksk = copy.deepcopy(Rykyk)
        Rykyqb = [copy.deepcopy(baseList) for _ in range(c.K)]
        Rykykmq = copy.deepcopy(Rykyqb)
        Rgkq = copy.deepcopy(Rykyqb)

        def _inner(x1, x2=None):
            if x2 is None:
                x2 = x1
            return x1 @ x2.T / c.Nonline
        
        allAlgos = list(zkq.keys()) + ['Centralized', 'Local', 'Unprocessed']
        msed = dict([
            (BFtype, np.zeros((c.nFrames, c.K))) for BFtype in allAlgos
        ])

        u = 0  # updating node index DANSE
        
        for l in range(c.nFrames):
            print(f"Frame {l + 1}/{c.nFrames}...", end='\r')
            # Compute signals for current frame
            y, s, d, n, latd, latn, sn =\
                self._get_latent_sigs(
                    nSamples=c.Nonline,  # <-- ONLY c.Nonline samples per channel
                    Amat=Amat,
                    Bmat=Bmat
                )

            for k in range(c.K):
                
                # Update the local SCM
                if c.upScmEveryNode or k == u:
                    Rykyk[k].update(_inner(y[k]))
                
                Lam = np.linalg.inv(Rykyk[k].val)

                for q in range(c.K):
                    if q == k:
                        continue
                    Qkq = QkqMat[k, q]  # number of common sources between k and q
                    # Update the k SCM with respect to incoming signal from q
                    if c.upScmEveryNode or k == u:
                        yqb = y[q][:Qkq, :]  # signal transmitted by q to k (Qkq x N)
                        Rykyqb[k][q].update(_inner(y[k], yqb))

                    # Update the k SCM of all contributions _but_ those of
                    # the common sources between k and q
                    gkq = get_gkq(k, q, Amat, Bmat, latd, latn, oMatd, oMatn)
                    # Rykykmq[k][q].update(_inner(y[k] - gkq))

                    if l % c.upEvery == 0:
                        Rykyqb_ = Rykyqb[k][q].val  # alias for conciseness
                        # Gam = np.linalg.inv(Rykykmq[k][q].val)
                        # Pkq[k][q]['LCMV'] = Gam @ Rykyqb_ @\
                        #     np.linalg.inv(Rykyqb_.T @ Gam @ Rykyqb_)  # (Mk x Qkq)
                        # Pkq[k][q]['LCMV'] /= np.linalg.norm(Pkq[k][q]['LCMV'])
                        Pkq[k][q]['Simple'] = Lam @ Rykyqb_
                        # Pkq[k][q]['LCMP'] = Lam @ Rykyqb_ @\
                        #     np.linalg.inv(Rykyqb_.T @ Lam @ Rykyqb_)  # (Mk x Qkq)
                        # Pkq[k][q]['LCMP'] /= np.linalg.norm(Pkq[k][q]['LCMP'])
                        if k == u:
                            Pkq[k][q]['DANSE'] = copy.deepcopy(tWk[k][:c.Mk, :])

                    # --- dMWF original definition ---
                    if c.upScmEveryNode or k == u:
                        Rgkq[k][q].update(_inner(gkq))
                    Ekloc = np.zeros((c.Mk, Qkq))
                    Ekloc[:Qkq, :] = np.eye(Qkq)
                    Pkq[k][q]['dMWF'] = Lam @ Rgkq[k][q].val @ Ekloc # (Mk x Qkq)
                    
                    pass
                    # Fused signals
                    for BFtype in zkq.keys():
                        zkq[BFtype]['y'][k][q] = Pkq[k][q][BFtype].T @ y[k]
                        zkq[BFtype]['s'][k][q] = Pkq[k][q][BFtype].T @ s[k]
                        if BFtype == 'DANSE':
                            if zkq[BFtype]['y'][k][q].shape[0] == 1:
                                pass
        
            # Compute target MWF at each node
            dhatk = dict([
                (BFtype, [None for _ in range(c.K)])
                for BFtype in allAlgos
            ])

            for k in range(c.K):
                for BFtype in zkq.keys():
                    RtyCurr = Rty[BFtype][k]
                    RtsCurr = Rts[BFtype][k]
                    ty = self.build_tilde(y[k], zkq[BFtype]['y'], k)
                    ts = self.build_tilde(s[k], zkq[BFtype]['s'], k)
                    # Update the tilde SCMs
                    if c.upScmEveryNode or k == u:
                        RtyCurr.update(_inner(ty))
                        RtsCurr.update(_inner(ts))
                    # Selection vector
                    tWkFull = np.linalg.inv(RtyCurr.val) @ RtsCurr.val 
                    tE = np.zeros((RtyCurr.val.shape[0], c.D))
                    tE[:c.D, :] = np.eye(c.D)
                    tWkCurr = tWkFull @ tE
                    if BFtype == 'DANSE' and k == u:
                        tE2 = np.zeros((RtyCurr.val.shape[0], c.Qd))
                        tE2[:c.Qd, :] = np.eye(c.Qd)
                        tWk[k] = tWkFull @ tE2  # store for DANSE
                    # Compute target signal estimate
                    dhatk[BFtype][k] = tWkCurr.T @ ty

            # Compute centralized and local MWFs
            yc = np.concatenate(y, axis=0)
            sc = np.concatenate(s, axis=0)
            # Update centralized SCMs
            Ryy.update(_inner(yc))
            Rss.update(_inner(sc))
            dhatk['Centralized'] = [None for _ in range(c.K)]
            dhatk['Local'] = [None for _ in range(c.K)]
            for k in range(c.K):
                Ek = np.zeros((Ryy.val.shape[0], c.D))
                Ek[k * c.Mk:k * c.Mk + c.D, :] = np.eye(c.D)
                Wk = np.linalg.inv(Ryy.val) @ Rss.val @ Ek
                dhatk['Centralized'][k] = Wk.T @ yc
                # Local MWF (Rykyk already updated in first k-loop)
                Rsksk[k].val = Rss.val[
                    k * c.Mk:(k + 1) * c.Mk,
                    k * c.Mk:(k + 1) * c.Mk
                ]
                assert np.allclose(Rykyk[k].val, Ryy.val[
                    k * c.Mk:(k + 1) * c.Mk,
                    k * c.Mk:(k + 1) * c.Mk
                ]), "Rykyk and Ryy[k...] are not equal!"
                Ekk = np.zeros((Rykyk[k].val.shape[0], c.D))
                Ekk[:c.D, :] = np.eye(c.D)
                Wkloc = np.linalg.inv(Rykyk[k].val) @ Rsksk[k].val @ Ekk
                dhatk['Local'][k] = Wkloc.T @ y[k]

            # Compute unprocessed MSEd
            dhatk['Unprocessed'] = [y[k][:c.D, :] for k in range(c.K)]  # unprocessed signal

            # Compute MSE_d for current frame
            for BFtype, dh in dhatk.items():
                if msed[BFtype] is None:
                    msed[BFtype][l, :] = [None for _ in range(c.K)]
                msed[BFtype][l, :] = [
                    np.mean(dh[k] - d[k]) ** 2
                    for k in range(c.K)
                ]
            
            # profiler.stop()
            # print(profiler.output_text(unicode=True, color=True))
            # pass

            u = (u + 1) % c.K  # update node index for DANSE

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
        s.Rsslat = np.diag(np.diag(latd @ latd.T / c.Nbatch))
        # s.Rnnlat = latn @ latn.T / c.N
        s.Rnnlat = np.diag(np.diag(latn @ latn.T / c.Nbatch))
        snall = np.concatenate(sn, axis=0)
        # s.Rvv = snall @ snall.T / c.N
        s.Rvv = np.diag(np.diag(snall @ snall.T / c.Nbatch))
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

def get_gkq(k, q, Amat, Bmat, latd, latn, oMatd, oMatn):
    """
    Compute the kq-pair-specific common sources contributions to the
    microphone signal yk.
    """
    iComkqd = np.where(oMatd[:, k] & oMatd[:, q])[0]
    iComkqn = np.where(oMatn[:, k] & oMatn[:, q])[0]
    skq = Amat[k][:, iComkqd] @ latd[iComkqd, :]
    nkq = Bmat[k][:, iComkqn] @ latn[iComkqn, :]
    return skq + nkq  # <-- used for `Rgkq`