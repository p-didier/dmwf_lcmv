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
class SCM:
    val: np.ndarray = None
    beta: float = 0.995  # Exponential averaging factor
    dim: tuple[int] = (1, 1)  # SCM dimension [dim x dim]

    def __post_init__(self):
        if self.val is None:
            # SCM not yet initialized. Do it now
            if isinstance(self.dim, int) or len(self.dim) == 1:
                if isinstance(self.dim, tuple):
                    self.dim = list(self.dim)[0]
                self.val = 1e-8 * np.eye(self.dim)
            else:
                if self.dim[0] >= self.dim[1]:
                    self.val = 1e-8 * np.eye(self.dim[0])
                    self.val = self.val[:, :self.dim[1]]
                elif self.dim[0] >= self.dim[1]:
                    self.val = 1e-8 * np.eye(self.dim[1])
                    self.val = self.val[:self.dim[0], :]
    
    def update(self, yyH: np.ndarray):
        """Update the spatial covariance matrix using exponential averaging."""
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
        n = [Bmat[k] @ latn + sn[k] for k in range(c.K)]  # noise sources contribution
        d = [s[k][:c.D, :] for k in range(c.K)]  # target signals
        y = [s[k] + n[k] for k in range(c.K)]  # microphone signals
        return y, s, d, n, latd, latn
        
    def build_tilde_dmwf(self, yk, zkqCurr, kCurr):
        """Build \\tilde{y}_k (dMWF or iDANSE)."""
        zmk = np.concatenate(
            [
                zkqCurr[q][kCurr]
                for q in range(self.cfg.K)
                if q != kCurr
            ],
            axis=0
        )
        return np.concatenate([yk, zmk], axis=0)
    
    def build_tilde_danse(self, yk, zqCurr, kCurr, ti=False):
        """Build \\tilde{y}_k (DANSE or TI-DANSE)."""
        if ti:
            fct = np.sum
        else:
            fct = np.concatenate
        zmk = fct(
            [
                zqCurr[q]
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
        if c.foss:
            oMatd = np.ones((c.Qd, c.K), dtype=int)
            oMatn = np.ones((c.Qn, c.K), dtype=int)
        else:
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
        y, s, d, n, latd, latn = self._get_latent_sigs(
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
            'dMWF': copy.deepcopy(baseDict),
            'iDANSE': copy.deepcopy(baseDict),
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
                Pkq[k][q]['dMWF'] = Lam @ Rykyqb

                # --- dMWF original definition ---
                if c.scmEst == 'theoretical':
                    Rgkq = self.scms.Rgkq[k][q]
                elif c.scmEst == 'batch':
                    Rgkq = gkq @ gkq.T / c.Nbatch
                Ekloc = np.zeros((c.Mk, Qkq))
                Ekloc[:Qkq, :] = np.eye(Qkq)
                Pkq[k][q]['iDANSE'] = np.linalg.inv(Rykyk) @ Rgkq @ Ekloc # (Mk x Qkq)
                
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
                ty = self.build_tilde_dmwf(y[k], zkq[BFtype]['y'], k)
                ts = self.build_tilde_dmwf(s[k], zkq[BFtype]['s'], k)
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
        def _gen_baselist(danse=False):
            return [
                [
                    np.random.randn(QkqMat[k, q], c.Nonline)
                    if not danse else
                    np.random.randn(c.Qd, c.Nonline)
                    for q in range(c.K)
                ]
                for k in range(c.K)
            ]
        baseDict = {
            'y': _gen_baselist(),
            's': _gen_baselist(),
            'n': _gen_baselist(),
        }
        baseDictDANSE = {
            'y': _gen_baselist(danse=True),
            's': _gen_baselist(danse=True),
            'n': _gen_baselist(danse=True),
        }
        zkq = {
            'dMWF': copy.deepcopy(baseDict),
            'dMWF_iter': copy.deepcopy(baseDict),
            'iDANSE': copy.deepcopy(baseDict),
        }
        zk = {
            # 'DANSE': copy.deepcopy(baseDictDANSE),
            # 'TI-DANSE': copy.deepcopy(baseDictDANSE),
        }
        DANSElikealgos = list(zk.keys())
        dMWFlikealgos = list(zkq.keys())
        allDistAlgos = DANSElikealgos + dMWFlikealgos
        # Node-pair-specific fusion matrices (for all
        # but DANSE-like algorithms)
        Pkq = [[dict([
            (BFtype, np.random.randn(c.Mk, QkqMat[k, q]))
            for BFtype in dMWFlikealgos
        ]) for q in range(c.K)] for k in range(c.K)]
        # DANSE-like fusion matrices, one per node
        Pk = [dict([
            (BFtype, np.random.randn(c.Mk, c.Qd))
            for BFtype in DANSElikealgos
        ]) for _ in range(c.K)]
        # Estimation filters \tilde{W}_k
        def tilde_filter_init(algo, k):
            if algo == 'DANSE':
                dims = (c.Mk + c.Qd * (c.K - 1), c.Qd)
            elif algo == 'TI-DANSE':
                dims = (c.Mk + c.Qd, c.Qd)
            else:
                dims = (c.Mk + int(np.sum(QkqMat[k, :]) - QkqMat[k, k]), c.Qd)
            return (algo, np.random.randn(*dims))
        tWk = [dict([
            tilde_filter_init(BFtype, k) for BFtype in allDistAlgos
        ]) for k in range(c.K)]
        # SCMs
        Ryy = SCM(dim=c.M, beta=c.beta)
        Rss = copy.deepcopy(Ryy)
        Rnn = copy.deepcopy(Ryy)
        Rty = dict()
        for BFtype in allDistAlgos:
            if BFtype == 'DANSE':
                Rty[BFtype] = [
                    SCM(dim=c.Mk + c.Qd * (c.K - 1), beta=c.beta)
                    for _ in range(c.K)
                ]
            elif BFtype == 'TI-DANSE':
                Rty[BFtype] = [
                    SCM(dim=c.Mk + c.Qd, beta=c.beta)
                    for _ in range(c.K)
                ]
            else:
                Rty[BFtype] = [
                    SCM(dim=c.Mk + int(np.sum(QkqMat[q, :]) - QkqMat[q, q]), beta=c.beta)
                    for q in range(c.K)
                ]
        Rts = copy.deepcopy(Rty)
        Rtn = copy.deepcopy(Rty)
        Rykyk = [SCM(dim=c.Mk, beta=c.beta) for _ in range(c.K)]
        Rykyqb = [
            [SCM(dim=(c.Mk, QkqMat[k, q]), beta=c.beta) for q in range(c.K)]
            for k in range(c.K)
        ]
        Rykzqtk = copy.deepcopy(Rykyqb)
        Rgkq = [copy.deepcopy(Rykyk) for _ in range(c.K)]

        def _inner(x1, x2=None):
            if x2 is None:
                x2 = x1
            return x1 @ x2.T / c.Nonline
        
        allAlgos = allDistAlgos + ['Centralized', 'Local', 'Unprocessed']
        msed = dict([
            (BFtype, np.zeros((c.nFrames, c.K))) for BFtype in allAlgos
        ])

        u = 0  # updating node index DANSE
        i = 0  # iteration index
        
        for l in range(c.nFrames):
            print(f"Frame {l + 1}/{c.nFrames}...", end='\r')
            # Compute signals for current frame
            y, s, d, n, latd, latn =\
                self._get_latent_sigs(
                    nSamples=c.Nonline,  # <-- ONLY c.Nonline samples per channel
                    Amat=Amat,
                    Bmat=Bmat
                )

            for k in range(c.K):
                
                # Update the local SCM
                if c.upScmEveryNode or k == u:
                    Rykyk[k].update(_inner(y[k]))
                
                Rykykinv = np.linalg.inv(Rykyk[k].val)
            
                # DANSE-like algo processing (not neighbor-specific)
                for BFtype in DANSElikealgos:
                    zk[BFtype]['y'][k] = Pk[k][BFtype].T @ y[k]
                    zk[BFtype]['s'][k] = Pk[k][BFtype].T @ s[k]
                    zk[BFtype]['n'][k] = Pk[k][BFtype].T @ n[k]

                for q in range(c.K):
                    if q == k:
                        continue
                    Qkq = QkqMat[k, q]  # number of common sources between k and q
                    # Update the k SCM with respect to incoming signal from q
                    if c.upScmEveryNode or k == u:
                        yqb = y[q][:Qkq, :]  # signal transmitted by q to k (Qkq x N)
                        Rykyqb[k][q].update(_inner(y[k], yqb))
                        Rykzqtk[k][q].update(_inner(y[k], zkq['dMWF_iter']['y'][q][k]))

                    # Update the k SCM of all contributions _but_ those of
                    # the common sources between k and q
                    gkq = get_gkq(k, q, Amat, Bmat, latd, latn, oMatd, oMatn)
                    # Rykykmq[k][q].update(_inner(y[k] - gkq))

                    Pkq[k][q]['dMWF'] = Rykykinv @ Rykyqb[k][q].val
                    Pkq[k][q]['dMWF_iter'] = Rykykinv @ Rykzqtk[k][q].val

                    # --- dMWF original definition ---
                    if c.upScmEveryNode or k == u:
                        Rgkq[k][q].update(_inner(gkq))
                    Ekloc = np.zeros((c.Mk, Qkq))
                    Ekloc[:Qkq, :] = np.eye(Qkq)
                    Pkq[k][q]['iDANSE'] = Rykykinv @ Rgkq[k][q].val @ Ekloc # (Mk x Qkq)
                    
                    # Fused signals
                    for BFtype in dMWFlikealgos:
                        zkq[BFtype]['y'][k][q] = Pkq[k][q][BFtype].T @ y[k]
                        zkq[BFtype]['s'][k][q] = Pkq[k][q][BFtype].T @ s[k]
                        zkq[BFtype]['n'][k][q] = Pkq[k][q][BFtype].T @ n[k]
        
            # Compute target MWF at each node
            dhatk = dict([
                (BFtype, [None for _ in range(c.K)])
                for BFtype in allAlgos
            ])

            # Compute estimation SCMs, filters, and desired signal estimates
            for k in range(c.K):
                for BFtype in allDistAlgos:
                    RtyCurr = Rty[BFtype][k]
                    RtsCurr = Rts[BFtype][k]
                    RtnCurr = Rtn[BFtype][k]
                    if BFtype in DANSElikealgos:
                        ty = self.build_tilde_danse(y[k], zk[BFtype]['y'], k)
                        ts = self.build_tilde_danse(s[k], zk[BFtype]['s'], k)
                        tn = self.build_tilde_danse(n[k], zk[BFtype]['n'], k)
                    else:
                        ty = self.build_tilde_dmwf(y[k], zkq[BFtype]['y'], k)
                        ts = self.build_tilde_dmwf(s[k], zkq[BFtype]['s'], k)
                        tn = self.build_tilde_dmwf(n[k], zkq[BFtype]['n'], k)
                    # Update the tilde SCMs
                    if c.upScmEveryNode or k == u:
                        RtyCurr.update(_inner(ty))
                        RtsCurr.update(_inner(ts))
                        RtnCurr.update(_inner(tn))
                    
                    def _full_filtup():
                        # Filter update
                        if fullrank(RtyCurr.val):
                            tWkFull = filtup(
                                RtyCurr.val,
                                RtnCurr.val,
                                gevd=c.gevd if BFtype in DANSElikealgos else False,  # TODO: addess dMWF and iDANSE formulation for GEVD...
                                gevdRank=c.Qd
                            )
                        else:
                            tWkFull = np.eye(RtyCurr.val.shape[0])
                        tE = np.zeros((RtyCurr.val.shape[0], c.Qd))
                        tE[:c.Qd, :] = np.eye(c.Qd)
                        tWk[k][BFtype] = tWkFull @ tE

                    if BFtype in DANSElikealgos:
                        if k == u and l % c.upEvery == 0:
                            _full_filtup()
                            # Update the fusion matrices for DANSE-like algorithms
                            if BFtype == 'DANSE':
                                Pk[k][BFtype] = tWk[k][BFtype][:c.Mk, :]
                            if BFtype == 'TI-DANSE':
                                Pk[k][BFtype] = tWk[k][BFtype][:c.Mk, :] @\
                                    np.linalg.inv(tWk[k][BFtype][c.Mk:, :])
                            i += 1
                            print(f"\nIteration {i}... ({BFtype} up. node {u + 1})")
                    else:
                        _full_filtup()

                    # Compute target signal estimate
                    dhatk[BFtype][k] = tWk[k][BFtype][:, :c.D].T @ ty

            # Compute centralized and local MWFs
            yc = np.concatenate(y, axis=0)
            sc = np.concatenate(s, axis=0)
            nc = np.concatenate(n, axis=0)
            # Update centralized SCMs
            Ryy.update(_inner(yc))
            Rss.update(_inner(sc))
            Rnn.update(_inner(nc))
            dhatk['Centralized'] = [None for _ in range(c.K)]
            dhatk['Local'] = [None for _ in range(c.K)]
            for k in range(c.K):
                Ek = np.zeros((Ryy.val.shape[0], c.D))
                Ek[k * c.Mk:k * c.Mk + c.D, :] = np.eye(c.D)
                # Centralized filter update
                if fullrank(Ryy.val):
                    Wk = filtup(
                        Ryy.val,
                        Rnn.val,
                        gevd=c.gevd,
                        gevdRank=c.Qd
                    ) @ Ek
                else:
                    Wk = np.eye(Ryy.val.shape[0])
                dhatk['Centralized'][k] = Wk.T @ yc
                # Local MWF (Rykyk already updated in first k-loop)
                RykykCurr = Ryy.val[
                    k * c.Mk:(k + 1) * c.Mk,
                    k * c.Mk:(k + 1) * c.Mk
                ]
                RskskCurr = Rss.val[
                    k * c.Mk:(k + 1) * c.Mk,
                    k * c.Mk:(k + 1) * c.Mk
                ]
                RnknkCurr = Rnn.val[
                    k * c.Mk:(k + 1) * c.Mk,
                    k * c.Mk:(k + 1) * c.Mk
                ]
                Ekk = np.zeros((RykykCurr.shape[0], c.D))
                Ekk[:c.D, :] = np.eye(c.D)
                # Local filter update
                if fullrank(RykykCurr):
                    Wkloc = filtup(
                        RykykCurr,
                        RnknkCurr,
                        gevd=c.gevd,
                        gevdRank=c.Qd
                    ) @ Ekk
                else:
                    Wkloc = np.eye(RykykCurr.shape[0])
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
            
            if l % c.upEvery == 0:
                u = (u + 1) % c.K  # update node index for DANSE-like algorithms

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

def filtup(Ryy, Rnn, gevd=False, gevdRank=1):
    """GEVD filter update for a single time- or frequency-line."""
    if gevd:
        try:
            sigma, Xmat = sla.eigh(Ryy, Rnn)
        except np.linalg.LinAlgError as error:
            raise error
        idx = np.flip(np.argsort(sigma))
        sigma = sigma[idx]
        Xmat = Xmat[:, idx]
        # Inverse-hermitian of `Xmat`
        Qmat = np.linalg.pinv(Xmat.T.conj())
        # GEVLs tensor - low-rank approximation is done here
        Dmat = np.zeros_like(Ryy)
        for r in range(gevdRank):
            Dmat[r, r] = np.squeeze(1 - 1 / sigma[r])
        # Compute filters
        return np.linalg.pinv(Qmat.T.conj()) @ Dmat @ Qmat.conj().T
    else:
        return np.linalg.inv(Ryy) @ (Ryy - Rnn)

def fullrank(M):
    """Check if a matrix is full rank."""
    return np.linalg.matrix_rank(M) == min(M.shape)