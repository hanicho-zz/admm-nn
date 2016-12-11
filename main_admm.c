#include "mlp.c"
#include "admm.c"

int main(int argc, char **argv) {
     int procs, rank;

     gsl_ieee_env_setup();
     gsl_rng_env_setup();

     MPI_Init(&argc, &argv);
     MPI_Comm_size(MPI_COMM_WORLD, &procs);
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

     if (argc != 4) {
	  if (rank == 0)
	       fprintf(stderr, "Usage: admm beta gamma warming\n");

	  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
     }

     double beta = atof(argv[1]);
     double gamma = atof(argv[2]);
     unsigned int warming = atoi(argv[3]);

     if (beta <= 0 || gamma <= 0) {
	  if (rank == 0)
	       fprintf(stderr, "Invalid range of beta, gamma: %f, %f\n", beta, gamma);

	  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
     }

     size_t T   = 60000;
     size_t V   = 10000;
     size_t IN  = 784;
     size_t OUT = 10;

     struct mlp *net = calloc(1, sizeof(struct mlp));
     short active[]  = {ANN_HARDSIG, ANN_HARDSIG};
     size_t config[] = {IN, 500, OUT};
     mlp_init(net, 2, ANN_HINGE, active, config);

     struct admm_learn *obs = calloc(1, sizeof(struct admm_learn));
     obs->samples = T;
     obs->input   = IN;
     obs->output  = OUT;
     obs->nodes   = procs;
     obs->root    = 0;
     obs->error   = GSL_POSINF;
     obs->maxit   = 500;
     obs->update  = 10;
     obs->warming = warming;
     obs->beta    = beta;
     obs->gamma   = gamma;

     if (rank == obs->root) {
          printf("SAMPLES = %zd\n", obs->samples);
          printf("NODES   = %d\n", obs->nodes);
          printf("MAXIT   = %zd\n", obs->maxit);
          printf("WARMING = %zd\n", obs->warming);
          printf("BETA    = %.17f\n", obs->beta);
          printf("GAMMA   = %.17f\n\n", obs->gamma);

	  gsl_matrix *Xt = gsl_matrix_calloc(T, IN);
	  gsl_matrix *Yt = gsl_matrix_calloc(T, OUT);
	  gsl_matrix *Xv = gsl_matrix_calloc(V, IN);
	  gsl_matrix *Yv = gsl_matrix_calloc(V, OUT);

	  mnist_read("MNIST/train-images.idx3-ubyte", "MNIST/train-labels.idx1-ubyte", Xt, Yt);
	  mnist_read("MNIST/t10k-images.idx3-ubyte", "MNIST/t10k-labels.idx1-ubyte", Xv, Yv);

	  mlp_print(net);

          admm_train(obs, net, Xt, Yt, Xv, Yv);

          gsl_matrix_free(Xt);
          gsl_matrix_free(Yt);
          gsl_matrix_free(Xv);
          gsl_matrix_free(Yv);
     } else {
          gsl_matrix *nil = gsl_matrix_calloc(1, 1);
          admm_train(obs, net, nil, nil, nil, nil);
          gsl_matrix_free(nil);
     }

     free(obs);
     mlp_free(net);

     return MPI_Finalize();
}
