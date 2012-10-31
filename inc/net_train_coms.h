#ifndef NET_TRAIN_COMS_H_
#define NET_TRAIN_COMS_H_

#define TRAIN_API_PREFIX __declspec(dllimport)

typedef struct {
	double grad;
	double MSE;
	int time;
	int epoch;
} learning_info;

TRAIN_API_PREFIX int net_send_info(learning_info *info);
TRAIN_API_PREFIX int net_receive_info(char * input_data);
TRAIN_API_PREFIX int net_start_sockets(unsigned short port);
TRAIN_API_PREFIX int net_stop_sockets();


#endif