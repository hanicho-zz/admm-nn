#ifndef ADMM_H
#define ADMM_H

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_permutation.h>
#include <mpi.h>
#include "mlp.h"

#define ADMM_MINEPS 1e-7
#define ADMM_MAXIT 10000

struct admm_learn {
     size_t samples;
     size_t input;
     size_t output;

     int nodes;
     int root;

     double cost;
     double error;
     double time;

     size_t it;
     size_t maxit;
     size_t update;
     size_t warming;

     size_t min_it;
     double min_time;
     double min_error;

     double beta;
     double gamma;
};

struct admm_node {
     int rank;

     size_t L;
     struct mlp *net;

     gsl_matrix *X;
     gsl_matrix *Y;

     gsl_matrix *lambda;
     gsl_matrix **As;
     gsl_matrix **Zs;
};

struct admm_min_params {
     short f;
     void (*h)(gsl_vector*, const gsl_vector*);
     double (*C)(const gsl_vector*, const gsl_vector*);
     double beta;
     double gamma;
     double lambda;
     double a;
     double Wa;
     double y;
};

void admm_node_init(struct admm_node *node, struct mlp *net);
void admm_node_print(struct admm_node *node);
void admm_node_free(struct admm_node *node);
void admm_learn_print(struct admm_learn *obs);

double admm_min_fn(const gsl_vector *zs, void *params);
double admm_argmin(size_t *it, double *time, double *error, struct admm_min_params *params, const double z);

void admm_inv(gsl_matrix *A_p, const gsl_matrix *A);
void admm_MP_pinv(gsl_matrix *A_p, const gsl_matrix *A);
void admm_weights(struct admm_learn *obs, struct admm_node *node, size_t l);
void admm_update(struct admm_learn *obs, struct admm_node *node);
void admm_train(struct admm_learn *obs, struct mlp *net, const gsl_matrix *Xt, const gsl_matrix *Yt, const gsl_matrix *Xv, const gsl_matrix *Yv);

#endif
