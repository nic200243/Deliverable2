#include <iostream>
#include <vector>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

#define N_ITERATIONS 10

using namespace std;


float random (float min, float max);

bool checkSym(const vector<float> & mat, const unsigned int _dim);
void matTranspose(const vector<float> & mat, vector<float> & MatTransposed, const unsigned int _dim);

bool checkSymOMP(const vector<float> & mat, const unsigned int _dim);
void matTransposeOMP(const vector<float> & mat, vector<float> & MatTransposed, const unsigned int _dim);


int main(int argc, char** argv){
	
#ifndef _OPENMP
    cout << "This program is not compiled with OpenMP." << endl;
#else
	srand(time(NULL));
	const unsigned int DIM[5] = {128, 256, 1024, 2048, 4096};
	double start, end;
	
	//SEQUENZIALE E OpenMP
	for (int dim=0; dim<5; dim++){
		cout << "--- MATRIX SIZE N="<< DIM[dim] << " ---" << endl;
		vector<float> M = vector<float>((DIM[dim]*DIM[dim]), 0.0);
	    vector<float> T = vector<float>((DIM[dim]*DIM[dim]), 0.0);
	    double sum_time_transpose_seq = 0.0;
	    
	    for (int i=0; i<DIM[dim]; i++) {
	        for (int j=0; j<DIM[dim]; j++) {
	            M[(i*DIM[dim])+j] = random(-100.0, 100.0);
	        }
	    }
	    
	    for(int iteration=0; iteration<N_ITERATIONS; iteration++){
			start = omp_get_wtime();
		    matTranspose(M, T, DIM[dim]);
		    end = omp_get_wtime();
			double time_transpose_seq = end-start;
			
			sum_time_transpose_seq += time_transpose_seq;
		}
		
		double avg_time_transpose_seq = sum_time_transpose_seq/N_ITERATIONS;
		double avg_bandwidth_transpose_seq = (8*(DIM[dim]*DIM[dim]))/avg_time_transpose_seq;
		cout << "---" << endl;
		cout << "Average elapsed execution times of serial implementations evaluated over " << N_ITERATIONS << " iterations." << endl;
		cout << "Avg time_transpose_seq = " << avg_time_transpose_seq << "	| ";
		cout << "avg_bandwidth_transpose_seq = " << avg_bandwidth_transpose_seq/1000000000 << " GB/s" << endl;
		cout << "---" << endl << endl << endl;
		
    	for (int num_threads=1; num_threads<=32; num_threads*=2) {
	    double sum_time_transpose_OMP = 0.0;
	    omp_set_num_threads(num_threads);
	    
	        for (int iteration=0; iteration<N_ITERATIONS; iteration++){	    
				start = omp_get_wtime();
			    matTransposeOMP(M, T, DIM[dim]);
			    end = omp_get_wtime();
				double time_transpose_OMP = end-start;
	        	
				sum_time_transpose_OMP += time_transpose_OMP;
			}
			
			double avg_time_transpose_OMP = sum_time_transpose_OMP/N_ITERATIONS;
			double avg_bandwidth_transposeOMP = (8*(DIM[dim]*DIM[dim]))/avg_time_transpose_OMP;
			double avg_speedup_transpose = avg_time_transpose_seq / avg_time_transpose_OMP;
        	double avg_efficiency_transpose = (avg_speedup_transpose / num_threads) * 100;
        	
			cout << "---" << endl;
			cout << "Average elapsed execution times of parallel implementations evaluated over " << N_ITERATIONS << " iterations using " << num_threads << " threads." << endl;
			cout << "Avg time_transposeOMP = " << avg_time_transpose_OMP << "	| ";
			cout << "avg_bandwidth_transposeOMP = " << avg_bandwidth_transposeOMP/1000000000 << " GB/s" << endl;
			cout << "Speedup transposeOMP = " << avg_speedup_transpose << endl;
			cout << "Efficiency transposeOMP = " << avg_efficiency_transpose << "%" << endl;
			cout << "---" << endl << endl;
		}
		
		cout << endl << endl << endl <<endl;
	}
	
#endif				

    return 0;
}



float random (float min, float max) {
    return (float) rand() / (float) RAND_MAX * (max - min) + min;
}


bool checkSym(const vector<float> & mat, const unsigned int _dim) {
	bool isSym = true;
    for (int i=0; i<_dim; i++) {
        for (int j=0; j<_dim; j++) {
            if (mat[(i*_dim)+j] != mat[(j*_dim)+i])
                isSym = false;
        }
    }
    return isSym;
}


void matTranspose(const vector<float> & mat, vector<float> & MatTransposed, const unsigned int _dim) {
    for (int i=0; i<_dim; i++) {
        for (int j=0; j<_dim; j++) {	
            MatTransposed[(i*_dim)+j] = mat[(j*_dim)+i];
        }
    }
}


bool checkSymOMP(const vector<float> & mat, const unsigned int _dim){
	bool isSym = true;
	#pragma omp parallel shared(isSym)
	{
		#pragma omp for collapse(2) reduction(&&:isSym) 
		for (int i=0; i<_dim; i++) {
	        for (int j=0; j<_dim; j++) {
	            if (mat[(i*_dim)+j] != mat[(j*_dim)+i])	
	                isSym = false;
	        }
	    }
	}
	return isSym;
}


void matTransposeOMP(const vector<float> & mat, vector<float> & MatTransposed, const unsigned int _dim){
	#pragma omp parallel
	{
		#pragma omp for collapse(2) 
		for (int i=0; i<_dim; i++) {
	        for (int j=0; j<_dim; j++) {	
	            MatTransposed[(i*_dim)+j] = mat[(j*_dim)+i];
	        }
		}
	}
}

