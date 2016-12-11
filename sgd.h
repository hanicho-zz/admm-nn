#ifndef SGD_H
#define SGD_H

#include "mlp.h"

struct sgd_learn {
     double tol;
     double cost;
     double error;
     double time;
     size_t maxit;
     size_t it;
     size_t batch;
     size_t cursor;
     size_t update;

     double alpha;
     double mu;
     double lambda1;
     double lambda2;

     struct mlp *net;

     gsl_matrix *Xt;
     gsl_matrix *Yt;
     gsl_matrix *Xv;
     gsl_matrix *Yv;

     size_t L;
     gsl_matrix **dWs;
     gsl_vector **dbs;
};

void sgd_learn_init(struct sgd_learn *obs, struct mlp *net);
void sgd_learn_free(struct sgd_learn *obs);
void sgd_learn_print(struct sgd_learn *obs);

void sgd_gradients(gsl_matrix **Gs, struct mlp *net, gsl_matrix **As, gsl_matrix *Y);
void sgd_backprop(struct sgd_learn *obs, gsl_matrix *X, gsl_matrix *Y);
void sgd_train(struct sgd_learn *obs);

#endif
