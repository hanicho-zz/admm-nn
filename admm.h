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

int get_cluster_memory_usage_kb(long* vmrss_per_process, long* vmsize_per_process, int root, int np) {
     long vmrss_kb;
     long vmsize_kb;
     int ret_code = get_memory_usage_kb(&vmrss_kb, &vmsize_kb);

     if (ret_code != 0) {
	  printf("Could not gather memory usage!\n");
	  return ret_code;
     }

     MPI_Gather(&vmrss_kb, 1, MPI_UNSIGNED_LONG,
		vmrss_per_process, 1, MPI_UNSIGNED_LONG,
		root, MPI_COMM_WORLD);

     MPI_Gather(&vmsize_kb, 1, MPI_UNSIGNED_LONG,
		vmsize_per_process, 1, MPI_UNSIGNED_LONG,
		root, MPI_COMM_WORLD);

     return 0;
}

int get_global_memory_usage_kb(long* global_vmrss, long* global_vmsize, int np) {
     long vmrss_per_process[np];
     long vmsize_per_process[np];
     int ret_code = get_cluster_memory_usage_kb(vmrss_per_process, vmsize_per_process, 0, np);

     if (ret_code != 0) {
	  return ret_code;
     }

     *global_vmrss = 0;
     *global_vmsize = 0;
     for (int i = 0; i < np; i++) {
	  *global_vmrss += vmrss_per_process[i];
	  *global_vmsize += vmsize_per_process[i];
     }

     return 0;
}

#endif
