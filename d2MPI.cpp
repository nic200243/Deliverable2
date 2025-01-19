#include <iostream>
#include <vector>
#include <ctime>
#include <mpi.h>

#define N_ITERATIONS 10

using namespace std;

float random (float min, float max);

bool checkSymMPI(const vector<float>& mat, const unsigned int _size);
void matTransposeMPI(const vector<float>& mat, vector<float>& MatTransposed, const unsigned int _size);


int main(int argc, char** argv){
	srand(time(NULL));
	MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	const unsigned int DIM[5] = {128, 256, 1024, 2048, 4096};
	double start, end;
	
	if(rank==0){
		cout << "<<<<<<<< NUMBER OF PROCESSES = " << size << " >>>>>>>>" << endl;
	}
	
	for (int dim=0; dim<5; dim++){
		
		if (rank==0){
			cout << "--- MATRIX SIZE N="<< DIM[dim] << " ---" << endl;
		}
		
		vector<float> M = vector<float>((DIM[dim]*DIM[dim]), 0.0);
		vector<float> T = vector<float>((DIM[dim]*DIM[dim]), 0.0);	
	
		if (rank==0){
			for (int i=0; i<DIM[dim]; i++) {
		        for (int j=0; j<DIM[dim]; j++) {
		            M[(i*DIM[dim])+j] = random(-100.0, 100.0);
		        }
		    }
		}	
		
		if (size>1){
			MPI_Bcast(M.data(), DIM[dim] * DIM[dim], MPI_FLOAT, 0, MPI_COMM_WORLD);
		}
		
		double sum_time_transposeMPI = 0.0;	
		
		for(int iteration=0; iteration<N_ITERATIONS; iteration++){
				start = MPI_Wtime();
			    matTransposeMPI(M, T, DIM[dim]);
			    end = MPI_Wtime();
			    
			    MPI_Bcast(T.data(), dim * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
				double time_transposeMPI = end-start;
				sum_time_transposeMPI += time_transposeMPI;
		}
		
		double avg_time_transposeMPI = sum_time_transposeMPI/N_ITERATIONS;
		double avg_bandwidth_transposeMPI = (8*(DIM[dim]*DIM[dim] + size*DIM[dim]*DIM[dim]))/avg_time_transposeMPI;
		if (rank==0){
			cout << "avg_time_transposeMPI = " << avg_time_transposeMPI << " s" << "	| ";
			cout << "avg_bandwidth_transposeMPI = " << avg_bandwidth_transposeMPI/1000000000 << " GB/s" << endl << endl;
		}
	}
	
	if (rank==0){
		cout << endl << endl;
	}
	
	MPI_Finalize();

    return 0;
}


float random (float min, float max) {
    return (float) rand() / (float) RAND_MAX * (max - min) + min;
}


bool checkSymMPI(const vector<float>& mat, const unsigned int _size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = _size / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? _size : start_row + rows_per_process;

    bool isSym = true;
    for (int i = start_row; i < end_row && isSym; i++) {
        for (int j = 0; j < _size; j++) {
            if (mat[i * _size + j] != mat[j * _size + i]) {
                isSym = false;
            }
        }
    }

    bool globalSym;
    MPI_Allreduce(&isSym, &globalSym, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

    return globalSym;
}


void matTransposeMPI(const vector<float>& mat, vector<float>& MatTransposed, const unsigned int _size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = _size / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? _size : start_row + rows_per_process;

    vector<float> localTransposed(_size*_size, 0.0f);
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < _size; j++) {
            localTransposed[j * _size + i] = mat[i * _size + j];
        }
    }

    MPI_Reduce(localTransposed.data(), MatTransposed.data(), _size * _size, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
}

