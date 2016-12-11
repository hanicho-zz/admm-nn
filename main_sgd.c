#include "mlp.c"
#include "sgd.c"

int main(int argc, char **argv) {
     gsl_ieee_env_setup();
     gsl_rng_env_setup();
     gsl_set_error_handler(NULL);

     if (argc != 2) {
          fprintf(stderr, "sgd alpha\n");
          return 1;
     }

     size_t T = 60000;
     size_t V = 10000;

     size_t IN = 784;
     size_t OUT = 10;

     gsl_matrix *Xt = gsl_matrix_calloc(T, IN);
     gsl_matrix *Yt = gsl_matrix_calloc(T, OUT);
     gsl_matrix *Xv = gsl_matrix_calloc(V, IN);
     gsl_matrix *Yv = gsl_matrix_calloc(V, OUT);

     mnist_read("MNIST/train-images.idx3-ubyte", "MNIST/train-labels.idx1-ubyte", Xt, Yt);
     mnist_read("MNIST/t10k-images.idx3-ubyte", "MNIST/t10k-labels.idx1-ubyte", Xv, Yv);

     struct mlp *net = calloc(1, sizeof(struct mlp));
     size_t config[] = {IN, 500, OUT};
     short active[] = {ANN_SIGMOID, ANN_SIGMOID};

     mlp_init(net, 2, ANN_QUADRATIC, active, config);
     mlp_print(net);

     struct sgd_learn *obs = calloc(1, sizeof(struct sgd_learn));
     sgd_learn_init(obs, net);

     obs->tol    = 0.0;
     obs->maxit  = 500;
     obs->batch  = 60000;
     obs->update = 10;
     obs->alpha  = strtod(argv[1], NULL);
     obs->Xt     = Xt;
     obs->Yt     = Yt;
     obs->Xv     = Xv;
     obs->Yv     = Yv;
     obs->error  = mlp_error(net, obs->Xv, obs->Yv);
     obs->cost   = mlp_cost(net, obs->Xv, obs->Yv);

     sgd_train(obs);

     sgd_learn_free(obs);
     mlp_free(net);
     gsl_matrix_free(Xt);
     gsl_matrix_free(Yt);
     gsl_matrix_free(Xv);
     gsl_matrix_free(Yv);

     return EXIT_SUCCESS;
}
