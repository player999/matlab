#include <mkl_net.h>
#include <stdlib.h>
#include <string.h>
#include "mex.h"
#include <net_train_coms.h>

double *MSE_log;
unsigned int *time_log;
char ip_address[255];
unsigned int port;

int train_callback(mkl_network *net){
	learning_info info;
	char in_message[1024];
	info.MSE = net->MSE;
	info.grad = 0;
	info.time = net->elpased_time;
	info.epoch = net->epoch;
	net_send_info(&info);
	mexPrintf("MSE: %e\tTime: %d\n", net->MSE, net->elpased_time);
	MSE_log[net->epoch - 1] = net->MSE;
	time_log[net->epoch - 1] = net->elpased_time;
	mexEvalString("drawnow;");
	if (utIsInterruptPending()){
		mexPrintf("Cought Ctrl+C. Wait a 5 sec.\n");
		mexEvalString("drawnow;");
		utSetInterruptPending(0);
		return -1;
	}
	if (net_receive_info(in_message)){
		if(!strcmp("{command:stop}", in_message)){
			return -1
		}
	}
	return 0;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
//Input variables
	double *weights;
	unsigned int *Hs;
	unsigned int *Ws;
	double *alphabet;
	double *targets;
	unsigned int num_layers;
	unsigned int num_threads;
	unsigned int set_size;
	double desired_error;
	unsigned int max_epochs;
	unsigned int max_time;
	double lambda;
	double sigma;
	unsigned int report_interval;
	unsigned int log_size[2]; 
	unsigned int net_size[2]; 
	mkl_network *net;
	
	mxArray *mMSE_log;
	mxArray *mtime_log;
	mxArray *mweights;
	double *new_weights;
	
//Body
	if(nrhs != 15){
		mexErrMsgTxt("Must be 15 input arguments!");
		return;
	}
	
	weights = (double *)mxGetData(prhs[0]);
	Hs = (unsigned int *)mxGetData(prhs[1]);
	Ws = (unsigned int *)mxGetData(prhs[2]);
	alphabet = (double *)mxGetData(prhs[3]);
	targets = (double *)mxGetData(prhs[4]);
	num_layers = (unsigned int)mxGetScalar(prhs[5]);
	num_threads = (unsigned int)mxGetScalar(prhs[6]);
	set_size = (unsigned int)mxGetScalar(prhs[7]);
	desired_error = (double)mxGetScalar(prhs[8]);
	max_epochs = (unsigned int)mxGetScalar(prhs[9]);
	max_time = (unsigned int)mxGetScalar(prhs[10]);
	lambda = (double)mxGetScalar(prhs[11]);
	sigma = (double)mxGetScalar(prhs[12]);
	report_interval = (unsigned int)mxGetScalar(prhs[13]);
	port = (unsigned int)mxGetScalar(prhs[14]);
	MSE_log = (double *)mxCalloc(max_epochs, sizeof(double));
	time_log = (unsigned int *)mxCalloc(max_time, sizeof(unsigned int));
	
	net = mkl_net_init(Hs, Ws, num_layers, weights, num_threads);
	net_start_sockets(port);
	mkl_net_train_scg(alphabet, targets, set_size, net, max_time, max_epochs,
						desired_error, report_interval, lambda, sigma, train_callback);
	net_stop_sockets();
	mexPrintf("Saving!\n");
	
	new_weights =  (double *)mxCalloc(net->total_weights, sizeof(double));
	memcpy((void *)new_weights, (void *)net->weights, sizeof(double) * net->total_weights);
	
	nlhs = 3;
	log_size[0] = net->epoch;
	log_size[1] = 1;
	
	net_size[0] = net->total_weights;
	net_size[1] = 1;
	
	mMSE_log = mxCreateNumericArray(2, log_size, mxDOUBLE_CLASS, mxREAL);
	mtime_log = mxCreateNumericArray(2, log_size, mxINT32_CLASS, mxREAL);
	mweights = mxCreateNumericArray(2, net_size, mxDOUBLE_CLASS, mxREAL);
	mxSetData(mMSE_log, MSE_log);
	mxSetData(mtime_log, time_log);
	mxSetData(mweights, new_weights);
	
	plhs[0] = mweights;
	plhs[1] = mMSE_log;
	plhs[2] = mtime_log;
	
	mkl_net_destroy(net);
}