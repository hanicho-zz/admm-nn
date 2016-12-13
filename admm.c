#include "admm.h"

void admm_node_init(struct admm_node *node, struct mlp *net) {
     node->L = net->L;
     node->net = net;
     node->lambda = calloc(net->L, sizeof(gsl_matrix));
     node->As = calloc(net->L, sizeof(gsl_matrix));
     node->Zs = calloc(net->L, sizeof(gsl_matrix));
     MPI_Comm_rank(MPI_COMM_WORLD, &node->rank);
}

void admm_node_print(struct admm_node *node) {
     printf("RANK    = %d\n", node->rank);
     printf("L_N     = %zd\n", node->X->size1);
     printf("L_XN    = %zd\n", node->X->size2);
     printf("L_YN    = %zd\n", node->Y->size2);
}

void admm_node_free(struct admm_node *node) {
     int l;
     for (l = 0; l < node->L; l++) {
	  gsl_matrix_free(node->As[l]);
	  gsl_matrix_free(node->Zs[l]);
     }

     gsl_matrix_free(node->lambda);
     free(node->As);
     free(node->Zs);
}

void admm_learn_print(struct admm_learn *obs) {
     printf("ITERATION = %zd\n",   obs->it);
     printf("COST      = %.17f\n", obs->cost);
     printf("ERROR     = %.17f\n", obs->error);
     printf("WALLTIME  = %.17f\n", obs->time);
     printf("MINIT     = %zd\n",   obs->min_it);
     printf("MINERROR  = %.17f\n", obs->min_time);
     printf("MINTIME   = %.17f\n", obs->min_error);
}

double admm_min_fn(const gsl_vector *zs, void *params) {
     struct admm_min_params *p = (struct admm_min_params*) params;
     void (*h)(gsl_vector*, const gsl_vector*) = p->h;
     double (*C)(const gsl_vector*, const gsl_vector*) = p->C;
     double beta = p->beta;
     double gamma = p->gamma;
     double lambda = p->lambda;
     double a = p->a;
     double Wa = p->Wa;
     double y = p->y;
     double z = gsl_vector_get(zs, 0);

     assert(zs->size == 1);
     assert(!!h != !!C);

     if (h) {
	  gsl_vector *hz = gsl_vector_calloc(1);
	  double cons_g;
	  double cons_b;

	  h(hz, zs);
	  cons_g = a - gsl_vector_get(hz, 0);
	  cons_g = gamma * cons_g * cons_g;

	  cons_b = z - Wa;
	  cons_b = beta * cons_b * cons_b;

	  gsl_vector_free(hz);

	  return cons_g + cons_b;
     } else if (C) {
	  gsl_vector *ys = gsl_vector_calloc(1);
	  double cons_b;

	  cons_b = z - Wa;
	  cons_b = beta * cons_b * cons_b;

	  gsl_vector_set(ys, 0, y);
	  cons_b += C(zs, ys);

	  cons_b += z*lambda;

	  gsl_vector_free(ys);

	  return cons_b;
     } else {
	  fprintf(stderr, "Activation or cost function not supplied.\n");
	  assert(0);
	  return EINVAL;
     }
}

double admm_argmin(size_t *it, double *time, double *error,
		   struct admm_min_params *params, const double z) {
     assert(!!params->h != !!params->C);

     *it = 0;
     *time = 0;
     *error = 0;

     if (params->h && params->f == ANN_HARDSIG) {
	  double beta = params->beta;
	  double gamma = params->gamma;
	  double a = params->a;
	  double Wa = params->Wa;
	  double z0, z1, zz;
	  double y0, y1, yz;
	  double ret;
	  gsl_vector *tmp = gsl_vector_calloc(1);

	  if (Wa < 0)
	       z0 = Wa;
	  else
	       z0 = 0;

	  if (Wa > 1)
	       z1 = Wa;
	  else
	       z1 = 1;

	  zz = (beta*Wa + gamma*a) / (beta + gamma);

	  gsl_vector_set(tmp, 0, z0);
	  y0 = admm_min_fn(tmp, params);
	  gsl_vector_set(tmp, 0, z1);
	  y1 = admm_min_fn(tmp, params);
	  gsl_vector_set(tmp, 0, zz);
	  yz = admm_min_fn(tmp, params);

	  gsl_vector_free(tmp);

	  ret = GSL_MIN(y0, y1);
	  ret = GSL_MIN(ret, yz);

	  if (ret == y0)
	       ret = z0;
	  else if (ret == y1)
	       ret = z1;
	  else
	       ret = zz;

	  return ret;
     } else if(params->C && params->f == ANN_HINGE) {
	  double beta = params->beta;
	  double y = params->y;
	  double lambda = params->lambda;
	  double Wa = params->Wa;
	  double z0, z1;
	  double y0, y1;
	  double ret;
	  gsl_vector *tmp = gsl_vector_calloc(1);

	  assert(y == 1 || y == 0);

	  z0 = (2*beta*Wa) / (lambda + 2*beta);

	  if (y == 0) {
	       z1 = (2*beta*Wa - lambda - 1) / 2*beta;
	  } else {
	       z1 = (2*beta*Wa - lambda + 1) / 2*beta;
	  }

	  gsl_vector_set(tmp, 0, z0);
	  y0 = admm_min_fn(tmp, params);
	  gsl_vector_set(tmp, 0, z1);
	  y1 = admm_min_fn(tmp, params);

	  gsl_vector_free(tmp);

	  ret = GSL_MIN(y0, y1);

	  if (ret == y0)
	       ret = z0;
	  else
	       ret = z1;

	  return ret;
     }

     int status;
     gsl_multimin_function F;
     F.n = 1;
     F.f = &admm_min_fn;
     F.params = params;

     gsl_vector *xs = gsl_vector_calloc(F.n);
     gsl_vector *ss = gsl_vector_calloc(F.n);

     gsl_vector_set(xs, 0, z);
     gsl_vector_set_all(ss, 1);

     double start = MPI_Wtime();

     const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
     gsl_multimin_fminimizer *s = gsl_multimin_fminimizer_alloc(T, F.n);
     gsl_multimin_fminimizer_set(s, &F, xs, ss);

     do {
	  *it += 1;
	  status = gsl_multimin_fminimizer_iterate(s);

	  if (status) {
	       fprintf(stderr, "Error in computing argmin: %d\n", status);
	       break;
	  }

	  *error = gsl_multimin_fminimizer_size(s);
	  status = gsl_multimin_test_size(*error, ADMM_MINEPS);
     } while (status == GSL_CONTINUE && *it < ADMM_MAXIT);

     double z_ = gsl_vector_get(s->x, 0);

     double end = MPI_Wtime();
     *time = end - start;

     gsl_vector_free(xs);
     gsl_vector_free(ss);
     gsl_multimin_fminimizer_free(s);

     return z_;
}

void admm_inv(gsl_matrix *A_p, const gsl_matrix *A) {
     assert(A->size1 == A->size2);
     assert(A_p->size1 == A->size1);
     assert(A_p->size2 == A->size2);

     size_t n = A->size1;

     gsl_matrix *B = gsl_matrix_calloc(n, n);
     gsl_permutation *p = gsl_permutation_alloc(n);
     int signum;

     gsl_permutation_init(p);
     gsl_matrix_memcpy(B, A);
     gsl_linalg_LU_decomp(B, p, &signum);
     gsl_linalg_LU_invert(B, p, A_p);

     gsl_permutation_free(p);
     gsl_matrix_free(B);
}

void admm_MP_pinv(gsl_matrix *A_p, const gsl_matrix *A) {
     assert(A->size1 == A_p->size2);
     assert(A->size2 == A_p->size1);

     gsl_matrix *B;
     size_t n = A->size1;
     size_t m = A->size2;
     size_t i;
     bool swap;

     if (m > n) {
	  B = gsl_matrix_calloc(m, n);
	  gsl_matrix_transpose_memcpy(B, A);
	  swap = true;

	  i = m;
	  m = n;
	  n = i;
     } else {
	  B = gsl_matrix_calloc(n, m);
	  gsl_matrix_memcpy(B, A);
	  swap = false;
     }

     gsl_matrix *V = gsl_matrix_calloc(m, m);
     gsl_matrix *S = gsl_matrix_calloc(m, n);
     gsl_vector_view s = gsl_matrix_diagonal(S);

     gsl_matrix *X = gsl_matrix_calloc(m, m);
     gsl_vector *work = gsl_vector_calloc(m);
     gsl_linalg_SV_decomp_mod(B, X, V, &s.vector, work);
     gsl_vector_free(work);
     gsl_matrix_free(X);

     double tau = ANN_MACHEPS * gsl_vector_max(&s.vector);
     gsl_matrix *S_p = S;

     for (i = 0; i < m; i++) {
	  double inv = gsl_vector_get(&s.vector, i);
	  inv = inv > tau ? 1.0 / inv : 0.0;
	  gsl_matrix_set(S_p, i, i, inv);
     }

     gsl_matrix *U = gsl_matrix_calloc(n, n);

     for (i = 0; i < B->size2; i++) {
	  gsl_vector_view c = gsl_matrix_column(B, i);
	  gsl_matrix_set_col(U, i, &c.vector);
     }

     gsl_matrix_free(B);
     B = gsl_matrix_calloc(m, n);
     gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, V, S_p, 0.0, B);

     if (swap)
	  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, U, B, 0.0, A_p);
     else
	  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, B, U, 0.0, A_p);

     gsl_matrix_free(B);
     gsl_matrix_free(U);
     gsl_matrix_free(S);
     gsl_matrix_free(V);
}

void admm_weights(struct admm_learn *obs, struct admm_node *node, size_t l) {
     struct mlp *net = node->net;
     const gsl_matrix *A = l == 0 ? node->X : node->As[l-1];
     const gsl_matrix *Z = node->Zs[l];
     gsl_matrix *W = net->layers[l]->W;

     gsl_matrix *ZA_t   = gsl_matrix_calloc(A->size2, Z->size2);
     gsl_matrix *AA_t   = gsl_matrix_calloc(A->size2, A->size2);
     gsl_matrix *tmp_ZA = gsl_matrix_calloc(ZA_t->size1, ZA_t->size2);
     gsl_matrix *tmp_AA = gsl_matrix_calloc(AA_t->size1, AA_t->size2);

     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, A, Z, 0.0, tmp_ZA);
     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, A, A, 0.0, tmp_AA);

     MPI_Allreduce(tmp_ZA->data, ZA_t->data,
		   ZA_t->size1*ZA_t->size2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     MPI_Allreduce(tmp_AA->data, AA_t->data,
		   AA_t->size1*AA_t->size2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

     admm_MP_pinv(tmp_AA, AA_t);

     gsl_matrix_free(tmp_ZA);
     gsl_matrix_free(AA_t);

     gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmp_AA, ZA_t, 0.0, W);

     gsl_matrix_free(tmp_AA);
     gsl_matrix_free(ZA_t);
}

void admm_update(struct admm_learn *obs, struct admm_node *node) {
     struct mlp *net = node->net;
     const gsl_matrix *X = node->X;
     const gsl_matrix *Y = node->Y;

     assert(X->size1 == Y->size1);
     assert(X->size2 == net->layers[0]->W->size1);
     assert(Y->size2 == net->layers[net->L-1]->W->size2);

     size_t n = X->size1;

     double start = MPI_Wtime();

     size_t l, i, j;
     for (l = 0; l < net->L-1; l++) {
	  const gsl_matrix *W = net->layers[l]->W;
	  const gsl_matrix *A = l == 0 ? X : node->As[l-1];
	  gsl_matrix *Z = node->Zs[l];

	  admm_weights(obs, node, l);

	  /* As */
	  gsl_matrix *W1 = net->layers[l+1]->W;
	  gsl_matrix *A1 = node->As[l];
	  gsl_matrix *Z1 = node->Zs[l+1];

	  gsl_matrix *tmp1 = gsl_matrix_calloc(n, W1->size1);
	  gsl_matrix *tmp2 = gsl_matrix_calloc(W1->size1, W1->size1);
	  gsl_matrix *tmp2_p = gsl_matrix_calloc(W1->size1, W1->size1);
	  gsl_matrix *tmp_HZ = gsl_matrix_calloc(n, W1->size1);
	  gsl_matrix *I = gsl_matrix_calloc(W1->size1, W1->size1);

	  for (i = 0; i < n; i++) {
	       gsl_vector_view tmp_hz = gsl_matrix_row(tmp_HZ, i);
	       gsl_vector_view tmp_z = gsl_matrix_row(Z, i);
	       net->layers[l]->h(&tmp_hz.vector, &tmp_z.vector);
	  }
	  gsl_matrix_scale(tmp_HZ, obs->gamma);

	  gsl_matrix_set_identity(I);
	  gsl_matrix_scale(I, obs->gamma);

	  gsl_blas_dgemm(CblasNoTrans, CblasTrans, obs->beta, Z1, W1, 0.0, tmp1);
	  gsl_matrix_add(tmp1, tmp_HZ);
	  gsl_blas_dgemm(CblasNoTrans, CblasTrans, obs->beta, W1, W1, 0.0, tmp2);
	  gsl_matrix_add(tmp2, I);

	  admm_inv(tmp2_p, tmp2);
	  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmp1, tmp2_p, 0.0, A1);

	  gsl_matrix_free(I);
	  gsl_matrix_free(tmp_HZ);
	  gsl_matrix_free(tmp2_p);
	  gsl_matrix_free(tmp2);
	  gsl_matrix_free(tmp1);

	  /* Zs */
	  struct admm_min_params *params = calloc(1, sizeof(struct admm_min_params));
	  gsl_matrix *tmp_WA = gsl_matrix_calloc(A->size1, W->size2);

	  params->f = net->layers[l]->activation;
	  params->h = net->layers[l]->h;
	  params->beta = obs->beta;
	  params->gamma = obs->gamma;
	  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, W, 0.0, tmp_WA);

	  for(i = 0; i < Z->size1; i++) {
	       for (j = 0; j < Z->size2; j++) {
		    size_t min_it;
		    double min_time;
		    double min_error;

		    params->a = gsl_matrix_get(A1, i, j);
		    params->Wa = gsl_matrix_get(tmp_WA, i, j);

		    double z_ = admm_argmin(&min_it, &min_time, &min_error,
					    params, gsl_matrix_get(Z, i, j));

		    obs->min_it += min_it;
		    obs->min_time += min_time;
		    obs->min_error += min_error;

		    gsl_matrix_set(Z, i, j, z_);
	       }
	  }

	  free(params);
	  gsl_matrix_free(tmp_WA);
     }

     const gsl_matrix *W = net->layers[net->L-1]->W;
     const gsl_matrix *A = net->L > 1 ? node->As[net->L-2] : X;
     gsl_matrix *Z = node->Zs[net->L-1];
     gsl_matrix *Lambda = node->lambda;

     admm_weights(obs, node, net->L-1);

     /* Zs */
     struct admm_min_params *params = calloc(1, sizeof(struct admm_min_params));
     gsl_matrix *tmp_WA = gsl_matrix_calloc(A->size1, W->size2);

     params->f = net->cost;
     params->C = net->C;
     params->beta = obs->beta;
     params->gamma = obs->gamma;
     gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, W, 0.0, tmp_WA);

     for(i = 0; i < Z->size1; i++) {
	  for (j = 0; j < Z->size2; j++) {
	       size_t min_it;
	       double min_time;
	       double min_error;

	       params->y = gsl_matrix_get(Y, i, j);
	       params->lambda = gsl_matrix_get(Lambda, i, j);
	       params->Wa = gsl_matrix_get(tmp_WA, i, j);

	       double z_ = admm_argmin(&min_it, &min_time, &min_error, params, gsl_matrix_get(Z, i, j));

	       obs->min_it += min_it;
	       obs->min_time += min_time;
	       obs->min_error += min_error;

	       gsl_matrix_set(Z, i, j, z_);
	  }
     }

     free(params);

     /* Lambda */
     if (obs->it > obs->warming) {
	  gsl_matrix *tmp = gsl_matrix_calloc(Z->size1, Z->size2);

	  gsl_matrix_memcpy(tmp, Z);
	  gsl_matrix_sub(tmp, tmp_WA);
	  gsl_matrix_scale(tmp, obs->beta);
	  gsl_matrix_add(Lambda, tmp);

	  gsl_matrix_free(tmp);
     }

     gsl_matrix_free(tmp_WA);

     double end = MPI_Wtime();

     obs->time = end - start;
     obs->it++;
}

void admm_train(struct admm_learn *obs, struct mlp *net,
		const gsl_matrix *Xt, const gsl_matrix *Yt,
		const gsl_matrix *Xv, const gsl_matrix *Yv) {
     int procs, rank;

     MPI_Comm_size(MPI_COMM_WORLD, &procs);
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

     if (rank == obs->root) {
	  assert(procs >= obs->nodes);
	  assert(obs->samples == Xt->size1);
	  assert(obs->input == Xt->size2);
	  assert(obs->output == Yt->size2);
	  assert(Xt->size1 == Yt->size1);
	  assert(Xv->size1 == Yv->size1);
	  assert(Xt->size2 == Xv->size2);
	  assert(Yt->size2 == Yv->size2);
	  assert(obs->maxit >= obs->warming);
	  assert(obs->beta > 0);
	  assert(obs->gamma > 0);
	  assert(Xt->size2 == Xt->tda);
	  assert(Yt->size2 == Yt->tda);
	  assert(Xv->size2 == Xv->tda);
	  assert(Yv->size2 == Yv->tda);
     }

     struct admm_node *node = calloc(1, sizeof(struct admm_node));
     admm_node_init(node, net);

     size_t l_n = obs->samples / obs->nodes;
     size_t left = obs->samples % obs->nodes;
     l_n = rank == obs->nodes-1 ? l_n + left : l_n;

     node->X = gsl_matrix_calloc(l_n, obs->input);
     node->Y = gsl_matrix_calloc(l_n, obs->output);

     MPI_Scatter(Xt->data,      l_n*obs->input, MPI_DOUBLE,
		 node->X->data, l_n*obs->input, MPI_DOUBLE,
		 obs->root,     MPI_COMM_WORLD);
     MPI_Scatter(Yt->data,      l_n*obs->output, MPI_DOUBLE,
		 node->Y->data, l_n*obs->output, MPI_DOUBLE,
		 obs->root,     MPI_COMM_WORLD);

     mlp_activations(node->As, net, node->X);
     node->lambda = gsl_matrix_calloc(node->X->size1, net->layers[net->L-1]->W->size2);

     int l;
     for (l = 0; l < net->L; l++) {
	  node->Zs[l] = gsl_matrix_calloc(node->X->size1, net->layers[l]->W->size2);
	  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,
			 l == 0 ? node->X : node->As[l-1],
			 net->layers[l]->W,
			 0.0, node->Zs[l]);
     }

     do {
	  admm_update(obs, node);

	  if (node->rank == obs->root &&
	      obs->it % obs->update == 0) {
#if ANN_DEBUG
	       printf("\nUpdating reconstruction error... ");
#endif
	       double start = MPI_Wtime();
	       obs->error = mlp_error(net, Xv, Yv);
	       obs->cost = mlp_cost(net, Xv, Yv);
	       double end = MPI_Wtime();
#if ANN_DEBUG
	       printf("%f seconds\n", end - start);
	       admm_learn_print(obs);
	       fflush(stdout);
#endif
	  }

	  if (obs->it % obs->update == 0) {
	       int id, np;
	       char processor_name[MPI_MAX_PROCESSOR_NAME];
	       char hostname[MPI_MAX_PROCESSOR_NAME];
	       int processor_name_len;

	       MPI_Comm_size(MPI_COMM_WORLD, &np);
	       MPI_Comm_rank(MPI_COMM_WORLD, &id);
	       MPI_Get_processor_name(processor_name, &processor_name_len);

	       printf("Number_of_processes=%03d, My_rank=%03d, processor_name=%5s\n",
		      np, id, processor_name);

	       long vmrss_per_process[np];
	       long vmsize_per_process[np];
	       get_cluster_memory_usage_kb(vmrss_per_process, vmsize_per_process, obs->root, np);

	       if (id == 0) {
		    for (int k = 0; k < np; k++) {
			 printf("Process %03d: VmRSS = %6ld KB, VmSize = %6ld KB\n",
				k, vmrss_per_process[k], vmsize_per_process[k]);
		    }
	       }

	       long global_vmrss, global_vmsize;
	       get_global_memory_usage_kb(&global_vmrss, &global_vmsize, np);
	       if (id == 0) {
		    printf("Global memory usage: VmRSS = %6ld KB, VmSize = %6ld KB\n",
			   global_vmrss, global_vmsize);
	       }
	  }
     } while (obs->it < obs->maxit);

     admm_node_free(node);
}
