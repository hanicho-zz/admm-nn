#ifndef MLP_H
#define MLP_H

#include "ann.h"

struct mlp_layer {
     short activation;
     void (*h)(gsl_vector*, const gsl_vector*);
     void (*dh)(gsl_vector*, const gsl_vector*);

     gsl_vector *b;
     gsl_matrix *W;
};

struct mlp {
     short cost;
     double (*C)(const gsl_vector*, const gsl_vector*);
     void (*dC)(gsl_vector*, const gsl_vector*, const gsl_vector*);

     gsl_rng *rng;
     size_t L;
     struct mlp_layer **layers;
};

void mlp_layer_init(struct mlp_layer *layer, short activation, size_t x, size_t y, gsl_rng *rng);
void mlp_layer_free(struct mlp_layer *layer);
void mlp_layer_print(const struct mlp_layer *layer);
void mlp_init(struct mlp *net, size_t l, short cost, short activations[], size_t config[]);
void mlp_free(struct mlp *net);
void mlp_print(const struct mlp *net);

void mlp_feed(gsl_matrix *A, const struct mlp_layer *layer, const gsl_matrix *X);
void mlp_activate(gsl_matrix *A, const struct mlp *net, const gsl_matrix *X);
void mlp_activations(gsl_matrix **As, const struct mlp *net, const gsl_matrix *X);
double mlp_cost(const struct mlp *net, const gsl_matrix *X, const gsl_matrix *Y);
double mlp_error(const struct mlp *net, const gsl_matrix *X, const gsl_matrix *Y);
bool mlp_check(const struct mlp *net);
size_t mlp_write(const struct mlp *net, const char file[]);
size_t mlp_read(struct mlp *net, const char file[]);

#endif
