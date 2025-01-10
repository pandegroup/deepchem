# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Library to run Jackhmmer from Python."""

from concurrent import futures
import glob
import logging
import os
import subprocess
from urllib import request
from typing import Any, Callable, Sequence, Mapping, Optional

class Jackhmmer:
    def __init__(
        self,
        *, # star indicates that all following arguments must be keyword
        binary_path: str = None,
        database_path: str,
        n_cpu: int = 8,
        n_iter: int = 1,
        e_value: float = 0.0001,
        z_value: Optional[int] = None,
        get_tblout: bool = False,
        filter_f1: float = 0.0005,
        filter_f2: float = 0.00005,
        filter_f3: float = 0.0000005,
        incdom_e: Optional[float] = None,
        dom_e: Optional[float] = None,
        num_streamed_chunks: Optional[int] = None, 
        streaming_callback: Optional[Callable[[int], None]] = None,
    ):
        """Initializes the Python Jackhmmer wrapper, an interative protein search
      program.

      Parameters
      ----------
      binary_path: str
          The path to the jackhmmer executable if jackhmmer is not
          installed in the environment.
      database_path: str
          The path to the jackhmmer database (FASTA format).
      n_cpu: int
          The number of CPUs to give Jackhmmer.
      n_iter: int
          The number of Jackhmmer iterations.
      e_value: float
          The E-value, see Jackhmmer docs for more details.
      z_value: int, optional
          The Z-value, see Jackhmmer docs for more details.
      get_tblout: bool
          Whether to save tblout string.
      filter_f1: float
          MSV and biased composition pre-filter, set to >1.0 to turn off.
      filter_f2: float
          Viterbi pre-filter, set to >1.0 to turn off.
      filter_f3: float
          Forward pre-filter, set to >1.0 to turn off.
      incdom_e: float, optional
          Domain e-value criteria for inclusion of domains in MSA/next
          round.
      dom_e: float, optional
          Domain e-value criteria for inclusion in tblout.
      num_streamed_chunks: int, optional
          Number of database chunks to stream over.
      streaming_callback: callable, optional
          Callback function run after each chunk iteration with
          the iteration number as argument.


      References
      ----------
      .. [1] http://eddylab.org/software/hmmer/Userguide.pdf

      Notes
      -----
      This class requires jackhmmer to be installed or the binary path specified.
      
      Works on linux only.
      """

        if binary_path is None:
            self.binary_path = subprocess.run(
                ['which', 'jackhmmer'],
                stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
        else:
            self.binary_path = binary_path
        if binary_path == '':
            raise ValueError(
                "Jackhmmer binary not found. Please install Jackhmmer. Try 'conda install -c bioconda hmmer'."
            )

        self.database_path = database_path
        self.num_streamed_chunks = num_streamed_chunks

        if (not os.path.exists(self.database_path) and num_streamed_chunks is None):
            logging.error("Could not find Jackhmmer database %s", database_path)
            raise ValueError(f"Could not find Jackhmmer database {database_path}")

        self.n_cpu = n_cpu
        self.n_iter = n_iter
        self.e_value = e_value
        self.z_value = z_value
        self.filter_f1 = filter_f1
        self.filter_f2 = filter_f2
        self.filter_f3 = filter_f3
        self.incdom_e = incdom_e
        self.dom_e = dom_e
        self.get_tblout = get_tblout
        self.streaming_callback = streaming_callback

    def _query_chunk(self, input_fasta_path: str,
                     database_path: str) -> Mapping[str, Any]:
        """Queries the database chunk using Jackhmmer."""
        query_tmp_dir = "/tmp"
        sto_path = os.path.join(query_tmp_dir, "output.sto")

        # The F1/F2/F3 are the expected proportion to pass each of the filtering
        # stages (which get progressively more expensive), reducing these
        # speeds up the pipeline at the expensive of sensitivity.  They are
        # currently set very low to make querying Mgnify run in a reasonable
        # amount of time.
        cmd_flags = [
            # Don't pollute stdout with Jackhmmer output.
            "-o",
            "/dev/null",
            "-A",
            sto_path,
            "--noali",
            "--F1",
            str(self.filter_f1),
            "--F2",
            str(self.filter_f2),
            "--F3",
            str(self.filter_f3),
            "--incE",
            str(self.e_value),
            # Report only sequences with E-values <= x in per-sequence output.
            "-E",
            str(self.e_value),
            "--cpu",
            str(self.n_cpu),
            "-N",
            str(self.n_iter),
        ]
        if self.get_tblout:
            tblout_path = os.path.join(query_tmp_dir, "tblout.txt")
            cmd_flags.extend(["--tblout", tblout_path])

        if self.z_value:
            cmd_flags.extend(["-Z", str(self.z_value)])

        if self.dom_e is not None:
            cmd_flags.extend(["--domE", str(self.dom_e)])

        if self.incdom_e is not None:
            cmd_flags.extend(["--incdomE", str(self.incdom_e)])
            
        cmd = ([self.binary_path] + cmd_flags + [input_fasta_path, database_path])

        cmd_str = " ".join(cmd)
        process = subprocess.Popen(cmd_str,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    shell=True)

        out, stderr = process.communicate()
        retcode = process.wait()

        if retcode:
            raise RuntimeError("Jackhmmer failed\nstderr:\n%s\n" %
                                stderr.decode("utf-8"))

        # Get e-values for each target name
        tbl = ""
        if self.get_tblout:
            with open(tblout_path) as f:
                tbl = f.read()

        with open(sto_path) as f:
            sto = f.read()

        raw_output = dict(
            sto=sto,
            tbl=tbl,
            n_iter=self.n_iter,
            e_value=self.e_value,
        )

        return raw_output

    def query(self, input_fasta_path: str) -> Sequence[Mapping[str, Any]]:
        """Queries the database using Jackhmmer.

            Parameters
            ----------
            input_fasta_path: str
                The path to the target fasta file.

            Returns
            -------
            chunk_output: list
                A list of dictionaries containing the raw output of each chunk.

            Examples
            --------
            >>> from deepchem.utils.jackhmmer import Jackhmmer
            >>> j = Jackhmmer(database_path='assets/test.fasta')
            >>> result = j.query("assets/sequence.fasta")

            """
        if self.num_streamed_chunks is None:
            return [self._query_chunk(input_fasta_path, self.database_path)]

        db_basename = os.path.basename(self.database_path)
        def db_remote_chunk(db_idx): return f"{self.database_path}.{db_idx}"
        def db_local_chunk(db_idx): return f"/tmp/ramdisk/{db_basename}.{db_idx}"

        # Remove existing files to prevent OOM
        for f in glob.glob(db_local_chunk("[0-9]*")):
            try:
                os.remove(f)
            except OSError:
                print(f"OSError while deleting {f}")

        # Download the (i+1)-th chunk while Jackhmmer is running on the i-th chunk
        with futures.ThreadPoolExecutor(max_workers=2) as executor:
            chunked_output = []
            for i in range(1, self.num_streamed_chunks + 1):
                # Copy the chunk locally
                if i == 1:
                    future = executor.submit(
                        request.urlretrieve,
                        db_remote_chunk(i),
                        db_local_chunk(i),
                    )
                if i < self.num_streamed_chunks:
                    next_future = executor.submit(
                        request.urlretrieve,
                        db_remote_chunk(i + 1),
                        db_local_chunk(i + 1),
                    )

                # Run Jackhmmer with the chunk
                future.result()
                chunked_output.append(
                    self._query_chunk(input_fasta_path, db_local_chunk(i)))

                # Remove the local copy of the chunk
                os.remove(db_local_chunk(i))
                future = next_future
                if self.streaming_callback:
                    self.streaming_callback(i)
        return chunked_output
