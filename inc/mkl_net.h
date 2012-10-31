#ifndef MKL_NET_H
#define MKL_NET_H

#define API_PREFIX __declspec(dllexport)

typedef struct{
	double *weights;
	double *delta;
	double *temp_error;
	double *slope;
	double MSE;
	double *input;
	unsigned int total_weights;
	unsigned int total_neurons;
	unsigned int *W;
	unsigned int *H;
	unsigned int count;
	unsigned int total_samples;
	double *output;
	double epsilon;
	unsigned int numl;
	unsigned int elpased_time;
	unsigned int epoch;
	unsigned int reason;
	unsigned int threads;

	double lambdak;
	double lambdab;
	double sigma;
	double sigmak;
	double MSE1;
	double MSE2;
	double deltak;
	double diffk;
	double npk;
	double npk2;
	double uk;
	double alphak;
	double betak;

	double *pk;
	double *rk;
	double *rkp1;
	double *pkp1;
	double *sk;
	double *weights2;
	double *error1;
	double *error2;
	
	int success;
	int context;
	unsigned int iter;
} mkl_network;


API_PREFIX mkl_network *mkl_net_init(unsigned int *Hs, unsigned int *Ws, 
			unsigned int num_layers, double* initial_weights, 
			unsigned int threads);

//API_PREFIX int mkl_net_free(mkl_network *net);
API_PREFIX int mkl_net_destroy(mkl_network *net);
API_PREFIX int mkl_net_train_scg(double *alphabet, double *target, 
	unsigned int set_size, mkl_network *net, unsigned int max_time, 
	unsigned int max_epochs, double min_error, unsigned int report_interval,
	double lambda, double sigma, int callback(mkl_network *));

/*API_PREFIX int mkl_net_train_batch(double *alphabet, double *target, 
	unsigned int set_size, mkl_network *net, unsigned int max_time, 
	unsigned int max_epochs, double min_error, unsigned int report_interval,
	unsigned int num_threads, int callback(mkl_network *));

API_PREFIX mkl_network *mkl_net_init(unsigned int *Hs, unsigned int *Ws, 
			unsigned int num_layers, double* initial_weights,
			double epsilon, unsigned int num_threads);

API_PREFIX int mkl_net_destroy(mkl_network *net);
*/



#endif
