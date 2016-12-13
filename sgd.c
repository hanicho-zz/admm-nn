#include "sgd.h"

void sgd_learn_init(struct sgd_learn *obs, struct mlp *net) {
     obs->net = net;
     obs->L = net->L;
     obs->dWs = calloc(obs->L, sizeof(gsl_matrix));
     obs->dbs = calloc(obs->L, sizeof(gsl_vector));

     int l;
     for (l = 0; l < obs->L; l++) {
	  obs->dWs[l] = gsl_matrix_calloc(net->layers[l]->W->size1,
					  net->layers[l]->W->size2);
	  obs->dbs[l] = gsl_vector_calloc(net->layers[l]->b->size);
     }
}

void sgd_learn_free(struct sgd_learn *obs) {
     int l;
     for (l = 0; l < obs->L; l++) {
	  gsl_matrix_free(obs->dWs[l]);
	  gsl_vector_free(obs->dbs[l]);
     }

     free(obs->dWs);
     free(obs->dbs);
     free(obs);
}

void sgd_learn_print(struct sgd_learn *obs) {
     printf("ITERATION   = %zd\n",   obs->it);
     printf("BATCHCURSOR = %zd\n",   obs->cursor);
     printf("COST        = %.17f\n", obs->cost);
     printf("ERROR       = %.17f\n", obs->error);
     printf("WALLTIME    = %.17f\n", obs->time);

     long vmrss, vmsize;

     get_memory_usage_kb(&vmrss, &vmsize);
     printf("Current memory usage: VmRSS = %6ld KB, VmSize = %6ld KB\n",
	    vmrss, vmsize);
}

void sgd_gradients(gsl_matrix **Gs, struct mlp *net, gsl_matrix **As, gsl_matrix *Y) {
     assert(As[net->L-1]->size1 == Y->size1);
     assert(net->layers[net->L-1]->W->size2 == Y->size2);

     int l, i;
     for (l = 0; l < net->L; l++) {
	  Gs[l] = gsl_matrix_calloc(Y->size1, As[l]->size2);

	  for (i = 0; i < As[l]->size1; i++) {
	       gsl_vector_view g = gsl_matrix_row(Gs[l], i);
	       gsl_vector_view a = gsl_matrix_row(As[l], i);

	       net->layers[l]->dh(&g.vector, &a.vector);
	  }
     }

     for (l = net->L-1; l >= 0; l--) {
	  if (l == net->L-1) {
	       for (i = 0; i < As[l]->size1; i++) {
		    gsl_vector_view g = gsl_matrix_row(Gs[l], i);
		    gsl_vector_view h = gsl_matrix_row(As[l], i);
		    gsl_vector_view y = gsl_matrix_row(Y, i);
		    gsl_vector *tmp = gsl_vector_calloc(g.vector.size);

		    net->dC(tmp, &h.vector, &y.vector);
		    gsl_vector_mul(&g.vector, tmp);

		    gsl_vector_free(tmp);
	       }
	  } else {
	       gsl_matrix *tmp = gsl_matrix_calloc(Y->size1, Gs[l]->size2);

	       gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Gs[l+1], net->layers[l+1]->W, 0.0, tmp);
	       gsl_matrix_mul_elements(Gs[l], tmp);

	       gsl_matrix_free(tmp);
	  }
     }
}

void sgd_backprop(struct sgd_learn *obs, gsl_matrix *X, gsl_matrix *Y) {
     struct mlp *net = obs->net;
     struct timespec start;
     clock_gettime(CLOCK_REALTIME, &start);

     assert(X->size1 == Y->size1);
     assert(X->size2 == net->layers[0]->W->size1);
     assert(Y->size2 == net->layers[net->L-1]->W->size2);

     gsl_matrix **As = calloc(net->L, sizeof(gsl_matrix));
     gsl_matrix **Gs = calloc(net->L, sizeof(gsl_matrix));
     size_t n = X->size1;

     mlp_activations(As, net, X);
     sgd_gradients(Gs, net, As, Y);

     int l;
     for (l = 0; l < net->L; l++) {
	  size_t x = net->layers[l]->W->size1;
	  size_t y = net->layers[l]->W->size2;
	  gsl_matrix *A = l == 0 ? X : As[l-1];

	  /* WEIGHTS */
	  gsl_matrix *AG = gsl_matrix_calloc(x, y);
	  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, A, Gs[l], 0.0, AG);
	  gsl_matrix_scale(AG, 1.0 / n);

	  if (obs->mu) {
	       gsl_matrix *tmp_M = obs->dWs[l];
	       gsl_matrix_scale(tmp_M, obs->mu);
	       gsl_matrix_add(AG, tmp_M);
	  }

	  if (obs->lambda1) {
	       gsl_matrix *tmp_L1 = obs->dWs[l];
	       gsl_matrix_memcpy(tmp_L1, net->layers[l]->W);

	       int i, j;
	       for (i = 0; i < x; i++) {
		    for (j = 0; j < y; j++) {
			 double v = gsl_matrix_get(tmp_L1, i, j);
			 v = v == 0 ? 0 : GSL_SIGN(v);

			 gsl_matrix_set(tmp_L1, i, j, obs->lambda1 * v);
		    }
	       }

	       gsl_matrix_sub(AG, tmp_L1);
	  }

	  if (obs->lambda2) {
	       gsl_matrix *tmp_L2 = obs->dWs[l];
	       gsl_matrix_memcpy(tmp_L2, net->layers[l]->W);
	       gsl_matrix_scale(tmp_L2, obs->lambda2);
	       gsl_matrix_sub(AG, tmp_L2);
	  }

	  gsl_matrix_scale(AG, obs->alpha);
	  gsl_matrix_sub(net->layers[l]->W, AG);
	  gsl_matrix_memcpy(obs->dWs[l], AG);
	  gsl_matrix_free(AG);

	  /* BIASES */
	  gsl_vector *g = gsl_vector_calloc(y);
	  gsl_vector *unit = gsl_vector_calloc(n);
	  gsl_vector_set_all(unit, 1.0);

	  int i;
	  for (i = 0; i < y; i++) {
	       gsl_vector_view c = gsl_matrix_column(Gs[l], i);
	       gsl_blas_ddot(&c.vector, unit, gsl_vector_ptr(g, i));
	  }

	  gsl_vector_free(unit);
	  gsl_vector_scale(g, 1.0 / n);

	  if (obs->mu) {
	       gsl_vector *tmp_m = obs->dbs[l];
	       gsl_vector_scale(tmp_m, obs->mu);
	       gsl_vector_add(g, tmp_m);
	  }

	  gsl_vector_scale(g, obs->alpha);
	  gsl_vector_sub(net->layers[l]->b, g);
	  gsl_vector_memcpy(obs->dbs[l], g);
	  gsl_vector_free(g);

	  gsl_matrix_free(Gs[l]);
	  if (l > 0)
	       gsl_matrix_free(As[l-1]);
     }

     free(Gs);
     free(As);

     struct timespec end;
     clock_gettime(CLOCK_REALTIME, &end);

     obs->time  = (double)((end.tv_sec + end.tv_nsec*1e-9) - (start.tv_sec + start.tv_nsec*1e-9));
     obs->it++;

     if (obs->it % obs->update == 0) {
#if ANN_DEBUG
	  printf("Updating reconstruction error... ");
#endif
	  clock_gettime(CLOCK_REALTIME, &start);
	  obs->error = mlp_error(net, obs->Xv, obs->Yv);
	  obs->cost  = mlp_cost(net, obs->Xv, obs->Yv);
	  clock_gettime(CLOCK_REALTIME, &end);
#if ANN_DEBUG
	  printf("%f seconds\n", (double)((end.tv_sec + end.tv_nsec*1e-9) - (start.tv_sec + start.tv_nsec*1e-9)));
	  fflush(stdout);
#endif
     }
}

void sgd_train(struct sgd_learn *obs) {
     assert(obs->Xt);
     assert(obs->Yt);
     assert(obs->Xv);
     assert(obs->Yv);
     assert(obs->tol <= obs->error);
     assert(obs->maxit > obs->it);
     assert(obs->Xt->size1 >= obs->batch);
     assert(obs->Xt->size1 == obs->Yt->size1);
     assert(obs->Xv->size1 == obs->Yv->size1);
     assert(obs->net->layers[0]->W->size1 == obs->Xt->size2);
     assert(obs->net->layers[0]->W->size1 == obs->Xv->size2);
     assert(obs->net->layers[obs->L-1]->W->size2 == obs->Yt->size2);
     assert(obs->net->layers[obs->L-1]->W->size2 == obs->Yv->size2);

#if ANN_DEBUG
     printf("TOLERANCE     = %.17f\n", obs->tol);
     printf("MAXITERATIONS = %zd\n",   obs->maxit);
     printf("BATCHSIZE     = %zd\n",   obs->batch);
     printf("ALPHA         = %.17f\n", obs->alpha);
     printf("MU            = %.17f\n", obs->mu);
     printf("LAMBDA1       = %.17f\n", obs->lambda1);
     printf("LAMBDA2       = %.17f\n", obs->lambda2);
     printf("\n");
#endif
     do {
	  obs->cursor = obs->batch == obs->Xt->size1 ?
	       0 :
	       gsl_rng_uniform_int(obs->net->rng, obs->Xt->size1 - obs->batch);

	  gsl_matrix_view Xb = gsl_matrix_submatrix(obs->Xt,
						    obs->cursor, 0,
						    obs->batch, obs->Xt->size2);
	  gsl_matrix_view Yb = gsl_matrix_submatrix(obs->Yt,
						    obs->cursor, 0,
						    obs->batch, obs->Yt->size2);

	  sgd_backprop(obs, &Xb.matrix, &Yb.matrix);
#if ANN_DEBUG
	  if (obs->it % obs->update == 0) {
	       sgd_learn_print(obs);
	       printf("\n");
	  }
#endif
     } while (obs->it < obs->maxit && obs->error > obs->tol);
}
