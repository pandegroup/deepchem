import numpy as np
from typing import Any, Optional, Tuple
from deepchem.models.torch_models.inceptionv3 import InceptionV3Model
from deepchem.data import Dataset


class DeepVariant(InceptionV3Model):
    """
    DeepVariant model for genomic variant calling using InceptionV3 architecture.

    This model extends DeepChem's InceptionV3Model to add variant calling
    capabilities.

    The model expects input data in the form of pileup images with 6 channels:
    - Base identity
    - Base quality
    - Mapping quality
    - Strand orientation
    - Variant support
    - Differences from reference
    """

    def __init__(self,
                 n_tasks: int = 3,
                 in_channels: int = 6,
                 learning_rate: float = 0.064,
                 dropout_rate: float = 0.2,
                 decay_rate: float = 0.94,
                 decay_steps: int = 2,
                 warmup_steps: int = 10000,
                 min_qual: float = 30.0,
                 **kwargs):
        """Initialize DeepVariant model."""
        super(DeepVariant, self).__init__(n_tasks=n_tasks,
                                          in_channels=in_channels,
                                          learning_rate=learning_rate,
                                          dropout_rate=dropout_rate,
                                          decay_rate=decay_rate,
                                          decay_steps=decay_steps,
                                          warmup_steps=warmup_steps,
                                          **kwargs)
        self.min_qual = min_qual

    def _calculate_genotype_quality(
            self, probabilities: np.ndarray) -> Tuple[int, str]:
        """
        Calculate genotype quality scores from probabilities.

        Parameters
        ----------
        probabilities : np.ndarray
            An array of probabilities for each genotype.

        Returns
        -------
        quality : int
            The calculated genotype quality score.
        genotype : str
            The genotype with the highest probability.
        """
        max_prob_idx = np.argmax(probabilities)
        genotypes = ['0/0', '0/1', '1/1']
        genotype = genotypes[max_prob_idx]

        sorted_probs = np.sort(probabilities)[::-1]
        if sorted_probs[0] == 0:
            quality = 0
        else:
            quality = int(-10 * np.log10(1 - sorted_probs[0]))

        return quality, genotype

    def _calculate_PL_scores(self, probabilities: np.ndarray) -> str:
        """
        Calculate Phred-scaled genotype likelihoods.

        Parameters
        ----------
        probabilities : np.ndarray
            An array of probabilities for each genotype.

        Returns
        -------
        str
            A comma-separated string of Phred-scaled genotype likelihoods.
        """
        pl_scores = [-10 * np.log10(p) if p > 0 else 999 for p in probabilities]
        pl_scores = list(np.array(pl_scores) - min(pl_scores))
        return ','.join(map(str, map(int, pl_scores)))

    def _write_vcf_header(self, vcf_file: Any, sample_name: str) -> None:
        """
        Write VCF header information.

        Parameters
        ----------
        vcf_file : Any
            The file object where the VCF header will be written.
        sample_name : str
            The name of the sample to be included in the VCF header.

        Returns
        -------
        None
        """
        vcf_file.write("##fileformat=VCFv4.2\n")
        vcf_file.write("##source=DeepChemVariant\n")

        # Write format descriptions
        vcf_file.write(
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        vcf_file.write(
            '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">\n'
        )
        vcf_file.write(
            '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">\n')
        vcf_file.write(
            '##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled genotype likelihoods">\n'
        )

        # Write FILTER descriptions
        vcf_file.write('##FILTER=<ID=PASS,Description="All filters passed">\n')
        vcf_file.write(
            f'##FILTER=<ID=LowQual,Description="Quality below {self.min_qual}">\n'
        )

        # Write INFO descriptions
        vcf_file.write(
            '##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">\n')

        # Write column headers
        vcf_file.write(
            f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_name}\n"
        )

    def call_variants(
        self,
        dataset: Dataset,
        output_path: str,
        sample_name: Optional[str] = None,
    ) -> str:
        """
        Call variants and write results to VCF.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing the variant information.
        output_path : str
            The path where the VCF file will be written.
        sample_name : Optional[str], optional
            The name of the sample, by default None.


        Returns
        -------
        str
            The path to the output VCF file.
        """
        if sample_name is None:
            sample_name = "SAMPLE"

        # Get predictions
        predictions = self.predict(dataset)
        variant_info = dataset.y

        # Write VCF
        with open(output_path, 'w') as vcf_file:
            self._write_vcf_header(vcf_file, sample_name)

            for pred, var in zip(predictions, variant_info):
                quality, genotype = self._calculate_genotype_quality(pred)
                pl_scores = self._calculate_PL_scores(pred)

                filter_status = "PASS" if quality >= self.min_qual else "LowQual"
                info = f"DP={var['depth']}"
                format_str = "GT:GQ:DP:PL"
                sample_data = f"{genotype}:{quality}:{var['depth']}:{pl_scores}"

                vcf_file.write(
                    f"{var['chrom']}\t{var['pos']}\t.\t{var['ref_allele']}\t"
                    f"{var['alt_allele']}\t{quality}\t{filter_status}\t{info}\t"
                    f"{format_str}\t{sample_data}\n")

        return output_path
