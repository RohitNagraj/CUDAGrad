import time

from benchmark.cudagrad2D_benchmark import CUDAGrad2DBenchmark
from benchmark.parameters import Parameters
from benchmark.micrograd_benchmark import MicrogradBenchmark

if __name__ == "__main__":
    start = time.time()
    benchmark = CUDAGrad2DBenchmark(Parameters.N_INPUTS, Parameters.N_OUTPUTS)
    benchmark.run(Parameters.DATASET_SIZE)
    print(f"Time taken by CUDAGrad 2D: {time.time() - start}\n\n")
    
    start = time.time()
    benchmark = MicrogradBenchmark(Parameters.N_INPUTS, Parameters.N_OUTPUTS)
    benchmark.run(Parameters.DATASET_SIZE)
    print(f"Time taken by Micrograd: {time.time() - start}")
    