from __future__ import annotations
import torch
# dqc depend
import dqc


class OrbitalOrthogonalizer(dqc.hamilton.orbconverter.OrbitalOrthogonalizer):

    def convert2(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Convert the last 2 dimensions of the matrix with shape (..., nao, nao)
        into the new orbital basis sets with shape (..., nao2, nao2)
        """
        self._orthozer: torch.tensor = self._orthozer.to("cuda")
        res = self._orthozer.transpose(
            -2, -1).conj() @ mat.to("cuda") @ self._orthozer
        return res


class IdentityOrbConverter(dqc.hamilton.orbconverter.IdentityOrbConverter):

    def convert2(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Convert the last 2 dimensions of the matrix with shape (..., nao, nao)
        into the new orbital basis sets with shape (..., nao2, nao2)
        """
        return mat.to("cuda")
