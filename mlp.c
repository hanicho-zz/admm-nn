#include "mlp.h"

void mlp_layer_init(struct mlp_layer *layer, short activation, size_t x, size_t y, gsl_rng *rng) {
     layer->activation = activation;
     layer->b = gsl_vector_calloc(y);
     layer->W = gsl_matrix_calloc(x, y);

     switch (activation) {
     case ANN_IDENTITY:
	  layer->h = &ann_identity;
	  layer->dh = &ann_didentity;
	  break;
     case ANN_SIGMOID:
	  layer->h = &ann_sigmoid;
	  layer->dh = &ann_dsigmoid;
	  break;
     case ANN_TANH:
	  layer->h = &ann_tanh;
	  layer->dh = &ann_dtanh;
	  break;
     case ANN_HARDSIG:
	  layer->h = &ann_hardsig;
	  layer->dh = &ann_dhardsig;
	  break;
     case ANN_RELU:
	  layer->h = &ann_relu;
	  layer->dh = &ann_drelu;
	  break;
     case ANN_SOFTMAX:
	  layer->h = &ann_softmax;
	  layer->dh = &ann_dsoftmax;
	  break;
     default:
	  fprintf(stderr, "Activation function not valid: %hi\n", activation);
	  exit(EXIT_FAILURE);
     }

     /* http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf */
     int i, j;
     for (i = 0; i < x; i++)
	  for (j = 0; j < y; j++)
	       gsl_matrix_set(layer->W, i, j,
	       		      gsl_ran_flat(rng,
	       				   -(sqrt(6.0) / sqrt(x + y)),
	       				    (sqrt(6.0) / sqrt(x + y))));
}

void mlp_layer_free(struct mlp_layer *layer) {
     gsl_vector_free(layer->b);
     gsl_matrix_free(layer->W);
     free(layer);
}

void mlp_layer_print(const struct mlp_layer *layer) {
     printf("Layer (x=%zd, y=%zd, s=%hi)\n", layer->W->size1, layer->W->size2, layer->activation);
}

void mlp_init(struct mlp *net, size_t L, short cost, short activations[], size_t config[]) {
     net->rng = gsl_rng_alloc(gsl_rng_mrg);
     net->L = L;
     net->cost = cost;
     net->layers = calloc(L, sizeof(struct mlp_layer));

     switch (cost) {
     case ANN_QUADRATIC:
          net->C = &ann_quadratic;
          net->dC = &ann_dquadratic;
          break;
     case ANN_ENTROPY:
          net->C = &ann_entropy;
          net->dC = &ann_dentropy;
          break;
     case ANN_HINGE:
          net->C = &ann_hinge;
          net->dC = &ann_dhinge;
          break;
     default:
	  fprintf(stderr, "Cost function not valid: %hi\n", cost);
	  exit(EXIT_FAILURE);
     }

     int l;
     for (l = 0; l < L; l++) {
	  net->layers[l] = calloc(1, sizeof(struct mlp_layer));
	  mlp_layer_init(net->layers[l], activations[l], config[l], config[l+1], net->rng);
     }
}

void mlp_free(struct mlp *net) {
     gsl_rng_free(net->rng);

     int i;
     for (i = 0; i < net->L; i++)
	  mlp_layer_free(net->layers[i]);

     free(net->layers);
     free(net);
}

void mlp_print(const struct mlp *net) {
     printf("MLP  (L=%zd C=%hi)\n", net->L, net->cost);

     int l;
     for (l = 0; l < net->L; l++)
	  mlp_layer_print(net->layers[l]);
}

void mlp_feed(gsl_matrix *A, const struct mlp_layer *layer, const gsl_matrix *X) {
     assert(X->size1 == A->size1);
     assert(layer->W->size1 == X->size2);
     assert(layer->W->size2 == A->size2);
     
     gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, X, layer->W, 0.0, A);

     int i;
     for (i = 0; i < A->size1; i++) {
          gsl_vector_view a = gsl_matrix_row(A, i);

          gsl_vector_add(&a.vector, layer->b);
          layer->h(&a.vector, &a.vector);
     }
}

void mlp_activate(gsl_matrix *A, const struct mlp *net, const gsl_matrix *X) {
     assert(X->size1 == A->size1);
     assert(net->layers[0]->W->size1 == X->size2);
     assert(net->layers[net->L-1]->W->size2 == A->size2);

     gsl_matrix **As = calloc(net->L, sizeof(gsl_matrix));
     mlp_activations(As, net, X);
     gsl_matrix_memcpy(A, As[net->L-1]);
     
     int l;
     for (l = 0; l < net->L; l++)
          gsl_matrix_free(As[l]);

     free(As);
}

void mlp_activations(gsl_matrix **As, const struct mlp *net, const gsl_matrix *X) {
     assert(net->layers[0]->W->size1 == X->size2);

     int l;
     for (l = 0; l < net->L; l++) {
          As[l] = gsl_matrix_calloc(X->size1, net->layers[l]->W->size2);
          mlp_feed(As[l], net->layers[l], l == 0 ? X : As[l-1]);
     }
}

double mlp_cost(const struct mlp *net, const gsl_matrix *X, const gsl_matrix *Y) {
     assert(X->size1 == Y->size1);
     assert(net->layers[0]->W->size1 == X->size2);
     assert(net->layers[net->L-1]->W->size2 == Y->size2);

     double error = 0.0;
     gsl_matrix *A = gsl_matrix_calloc(Y->size1, Y->size2);

     mlp_activate(A, net, X);

     int i;
     for (i = 0; i < X->size1; i++) {
          gsl_vector_const_view a = gsl_matrix_const_row(A, i);
          gsl_vector_const_view y = gsl_matrix_const_row(Y, i);

          error += net->C(&a.vector, &y.vector);
     }

     error /= Y->size1;
     gsl_matrix_free(A);

     return error;
}

double mlp_error(const struct mlp *net, const gsl_matrix *X, const gsl_matrix *Y) {
     assert(X->size1 == Y->size1);
     assert(net->layers[0]->W->size1 == X->size2);
     assert(net->layers[net->L-1]->W->size2 == Y->size2);

     double error = 0.0;
     gsl_matrix *A = gsl_matrix_calloc(Y->size1, Y->size2);
     
     mlp_activate(A, net, X);

     int i;
     for (i = 0; i < X->size1; i++) {
          gsl_vector_const_view a = gsl_matrix_const_row(A, i);
          gsl_vector_const_view y = gsl_matrix_const_row(Y, i);

          int j = gsl_vector_max_index(&a.vector);
          int k = gsl_vector_max_index(&y.vector);

          if (j != k)
               error++;
     }

     error /= Y->size1;
     gsl_matrix_free(A);

     return error;
}

bool mlp_check(const struct mlp *net) {
     bool ret = 1;

     int l;
     for (l = 0; l < net->L; l++) {
	  double b_max = gsl_vector_max(net->layers[l]->b);
	  double W_max = gsl_matrix_max(net->layers[l]->W);

	  if (!gsl_finite(b_max) || !gsl_finite(W_max)) {
	       ret = 0;
	       break;
	  }
     }

     return ret;
}

size_t mlp_write(const struct mlp *net, const char file[]) {
     FILE *f = fopen(file, "wb");

     if (f == NULL) {
	  fprintf(stderr, "Unable to open file %s.\n", file);
	  exit(1);
     }

     long long head = 0xDEADBEEF;
     fwrite(&head, sizeof(long long), 1, f);
     fwrite(&net->L, sizeof(net->L), 1, f);
     fwrite(&net->cost, sizeof(net->cost), 1, f);

     int l;
     for (l = 0; l < net->L; l++)
	  fwrite(&net->layers[l]->activation, sizeof(net->layers[l]->activation), 1, f);

     for (l = 0; l < net->L; l++) {
	  if (l == 0)
	       fwrite(&net->layers[l]->W->size1, sizeof(net->layers[l]->W->size1), 1, f);

	  fwrite(&net->layers[l]->W->size2, sizeof(net->layers[l]->W->size2), 1, f);
     }

     for (l = 0; l < net->L; l++) {
	  gsl_vector_fwrite(f, net->layers[l]->b);
	  gsl_matrix_fwrite(f, net->layers[l]->W);
     }

     size_t len = ftell(f);
     fclose(f);

     return len;
}

size_t mlp_read(struct mlp *net, const char file[]) {
     FILE *f = fopen(file, "rb");

     if (f == NULL) {
     	  fprintf(stderr, "Unable to open file: %s\n", file);
     	  exit(1);
     }

     long long head;
     fread(&head, sizeof(long long), 1, f);

     if (head != 0xDEADBEEF) {
     	  fprintf(stderr, "Incorrect format in file: %s\n", file);
     	  exit(1);
     }

     size_t L;
     short cost;
     fread(&L, sizeof(size_t), 1, f);
     fread(&cost, sizeof(short), 1, f);

     size_t *config = calloc(L+1, sizeof(size_t));
     short *activations = calloc(L, sizeof(size_t));

     int l;
     for (l = 0; l < L; l++)
	  fread(&activations[l], sizeof(short), 1, f);
     
     for (l = 0; l < L+1; l++)
          fread(&config[l], sizeof(size_t), 1, f);
     
     mlp_init(net, L, cost, activations, config);
     
     free(config);
     free(activations);

     for (l = 0; l < L; l++) {
     	  gsl_vector_fread(f, net->layers[l]->b);
     	  gsl_matrix_fread(f, net->layers[l]->W);
     }

     size_t len = ftell(f);
     fclose(f);

     return len;
}
