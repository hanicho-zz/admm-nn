#ifndef ANN_H
#define ANN_H

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_ieee_utils.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_vector.h>

/* #define GSL_IEEE_MODE "extended-precision" */
/* FLT_EVAL_METHOD 2/0 */
/* Condition all s and C's using a MACHEPS */

#define ANN_QUADRATIC 1
#define ANN_ENTROPY   2
#define ANN_HINGE     3

#define ANN_IDENTITY 1
#define ANN_SIGMOID  2
#define ANN_TANH     3
#define ANN_HARDSIG  4
#define ANN_RELU     5
#define ANN_SOFTMAX  6

#define ANN_DEBUG   1
#define ANN_MACHEPS 1e-15

double ann_quadratic(const gsl_vector *a, const gsl_vector *y) {
     assert(a->size == y->size);

     double cost = 0.0;

     int i;
     for (i = 0; i < a->size; i++) {
          double v = gsl_vector_get(a, i) - gsl_vector_get(y, i);
          cost += v*v;
     }

     cost /= 2.0;

     return cost;
}
void ann_dquadratic(gsl_vector *e, const gsl_vector *a, const gsl_vector *y) {
     assert(e->size == a->size);
     assert(a->size == y->size);

     gsl_vector_memcpy(e, a);
     gsl_vector_sub(e, y);
}

double ann_entropy(const gsl_vector *a, const gsl_vector *y) {
     assert(a->size == y->size);

     double cost = 0.0;

     int i;
     for (i = 0; i < a->size; i++) {
          double av = gsl_vector_get(a, i);
          double yv = gsl_vector_get(y, i);

          cost += yv * gsl_sf_log(av) +
            (1.0-yv) * gsl_sf_log(1.0-av);
     }

     cost = -cost;

     return cost;
}
void ann_dentropy(gsl_vector *e, const gsl_vector *a, const gsl_vector *y) {
     assert(e->size == a->size);
     assert(a->size == y->size);

     gsl_vector *div = gsl_vector_calloc(e->size);
     gsl_vector_memcpy(div, a);
     gsl_vector_scale(div, -1.0);
     gsl_vector_add_constant(div, 1.0);
     gsl_vector_mul(div, a);

     gsl_vector_memcpy(e, a);
     gsl_vector_sub(e, y);
     gsl_vector_div(e, div);

     gsl_vector_free(div);
}

double ann_hinge(const gsl_vector *a, const gsl_vector *y) {
     assert(a->size == y->size);

     double cost = 0.0;

     int i;
     for (i = 0; i < a->size; i++) {
          if (gsl_vector_get(y, i) == 0) {
               cost += GSL_MAX(0, gsl_vector_get(a, i));
          } else if (gsl_vector_get(y, i) == 1) {
               cost += GSL_MAX(0, 1.0 - gsl_vector_get(a, i));
          } else {
               fprintf(stderr, "Not a binary classification.\n");
          }
     }

     return cost;
}

void ann_dhinge(gsl_vector *e, const gsl_vector *a, const gsl_vector *y) {
     assert(0);
}

void ann_identity(gsl_vector *y, const gsl_vector *x) {
     assert(x->size == y->size);
     gsl_vector_memcpy(y, x);
}
void ann_didentity(gsl_vector *y, const gsl_vector *x) {
     assert(x->size == y->size);
     gsl_vector_set_all(y, 1.0);
}

void ann_sigmoid(gsl_vector *y, const gsl_vector *x) {
     assert(x->size == y->size);
     gsl_vector_memcpy(y, x);

     int i;
     for (i = 0; i < y->size; i++) {
          double v = gsl_vector_get(y, i);
          gsl_vector_set(y, i, 1.0 / (1.0 + gsl_sf_exp(-v)));
     }
}
void ann_dsigmoid(gsl_vector *y, const gsl_vector *x) {
     assert(x->size == y->size);
     gsl_vector_memcpy(y, x);

     int i;
     for (i = 0; i < y->size; i++) {
          double v = gsl_vector_get(y, i);
          gsl_vector_set(y, i, v * (1.0 - v));
     }
}

void ann_tanh(gsl_vector *y, const gsl_vector *x) {
     assert(x->size == y->size);
     gsl_vector_memcpy(y, x);

     int i;
     for (i = 0; i < y->size; i++) {
          double v = gsl_vector_get(y, i);
          gsl_vector_set(y, i, tanh(v));
     }
}
void ann_dtanh(gsl_vector *y, const gsl_vector *x) {
     assert(x->size == y->size);
     gsl_vector_memcpy(y, x);

     int i;
     for (i = 0; i < y->size; i++) {
          double v = gsl_vector_get(y, i);
          gsl_vector_set(y, i, 1.0 - v*v);
     }
}

void ann_hardsig(gsl_vector *y, const gsl_vector *x) {
     assert(x->size == y->size);
     gsl_vector_memcpy(y, x);

     int i;
     for (i = 0; i < y->size; i++) {
          double v = gsl_vector_get(y, i);
          v = v < 0 ? 0 : v;
          v = v > 1 ? 1 : v;

          gsl_vector_set(y, i, v);
     }
}
void ann_dhardsig(gsl_vector *y, const gsl_vector *x) {
     assert(x->size == y->size);
     gsl_vector_memcpy(y, x);

     int i;
     for (i = 0; i < y->size; i++) {
          double v = gsl_vector_get(y, i);
          v = v < 0 ? 0 : v;
          v = v > 1 ? 0 : v;
          v = v == 0 ? 0 : 1;

          gsl_vector_set(y, i, v);
     }
}

void ann_relu(gsl_vector *y, const gsl_vector *x) {
     assert(x->size == y->size);
     gsl_vector_memcpy(y, x);

     int i;
     for (i = 0; i < y->size; i++) {
          double v = gsl_vector_get(y, i);
          gsl_vector_set(y, i, v < 0 ? 0 : v);
     }
}
void ann_drelu(gsl_vector *y, const gsl_vector *x) {
     assert(x->size == y->size);
     gsl_vector_memcpy(y, x);

     int i;
     for (i = 0; i < y->size; i++) {
          double v = gsl_vector_get(y, i);
          gsl_vector_set(y, i, v < 0 ? 0 : 1);
     }
}

void ann_softmax(gsl_vector *y, const gsl_vector *x) {
     assert(x->size == y->size);
     gsl_vector_memcpy(y, x);

     int i;
     for (i = 0; i < y->size; i++) {
          double v = gsl_vector_get(y, i);
	  gsl_vector_set(y, i, gsl_sf_exp(v));
     }

     double sum = gsl_blas_dasum(y);
     gsl_blas_dscal(1.0 / sum, y);
}
void ann_dsoftmax(gsl_vector *y, const gsl_vector *x) {
     assert(x->size == y->size);
     gsl_vector_memcpy(y, x);

     int i;
     for (i = 0; i < y->size; i++) {
          double v = gsl_vector_get(y, i);
	  gsl_vector_set(y, i, v * (1.0 - v));
     }
}

size_t samples_write(char file[], gsl_matrix *X, gsl_matrix *Y) {
     assert(X->size1 == Y->size1);

     FILE *f = fopen(file, "wb");

     if (f == NULL) {
	  fprintf(stderr, "Unable to open file: %s\n", file);
	  exit(EXIT_FAILURE);
     }

     long long head = 0xCAFEBABE;
     fwrite(&head, sizeof(long long), 1, f);

     fwrite(&X->size1, sizeof(size_t), 1, f);
     fwrite(&X->size2, sizeof(size_t), 1, f);
     fwrite(&Y->size1, sizeof(size_t), 1, f);
     fwrite(&Y->size2, sizeof(size_t), 1, f);

     gsl_matrix_fwrite(f, X);
     gsl_matrix_fwrite(f, Y);

     size_t len = ftell(f);
     fclose(f);

     return len;
}

void samples_read(char file[], gsl_matrix *X, gsl_matrix *Y) {
     assert(X->size1 == Y->size1);

     FILE *f = fopen(file, "rb");

     if (f == NULL) {
	  fprintf(stderr, "Unable to open file: %s\n", file);
	  exit(EXIT_FAILURE);
     }

     long long head;
     fread(&head, sizeof(long long), 1, f);

     if (head != 0xCAFEBABE) {
	  fprintf(stderr, "Incorrect format in file: %s\n", file);
	  exit(EXIT_FAILURE);
     }

     size_t x1;
     size_t x2;
     size_t y1;
     size_t y2;

     fread(&x1, sizeof(size_t), 1, f);
     fread(&x2, sizeof(size_t), 1, f);
     fread(&y1, sizeof(size_t), 1, f);
     fread(&y2, sizeof(size_t), 1, f);

     if (X->size1 != x1 || X->size2 != x2 || Y->size1 != y1 || Y->size2 != y2) {
	  fprintf(stderr, "Malformed data in file: %s\n", file);
	  exit(EXIT_FAILURE);
     }

     gsl_matrix_fread(f, X);
     gsl_matrix_fread(f, Y);

     fclose(f);
}

void mnist_read(char images[], char labels[], gsl_matrix *X, gsl_matrix *Y) {
     assert(X->size1 == Y->size1);
     assert(Y->size2 == 10);

     FILE *fi = fopen(images, "rb");
     FILE *fl = fopen(labels, "rb");

     if (fi == NULL || fl == NULL) {
	  fprintf(stderr, "Unable to open files: %s, %s\n", images, labels);
	  exit(EXIT_FAILURE);
     }

     unsigned char i_head[4];
     unsigned char l_head[4];

     fread(&i_head, 1, 4, fi);
     fread(&l_head, 1, 4, fl);

     if (i_head[0] != 0x00 || i_head[1] != 0x00 || i_head[2] != 0x08 || i_head[3] != 0x03 ||
	 l_head[0] != 0x00 || l_head[1] != 0x00 || l_head[2] != 0x08 || l_head[3] != 0x01) {
	  fprintf(stderr, "Incorrect format in files: %s, %s\n", images, labels);
     }

     int32_t i_n = 0;
     int32_t l_n = 0;
     int32_t i_x = 0;
     int32_t i_y = 0;

     fread(&i_head, 1, 4, fi);
     fread(&l_head, 1, 4, fl);

     i_n = i_head[0];
     i_n <<= 8;
     i_n |= i_head[1];
     i_n <<= 8;
     i_n |= i_head[2];
     i_n <<= 8;
     i_n |= i_head[3];

     l_n = l_head[0];
     l_n <<= 8;
     l_n |= l_head[1];
     l_n <<= 8;
     l_n |= l_head[2];
     l_n <<= 8;
     l_n |= l_head[3];

     fread(&i_head, 1, 4, fi);

     i_x = i_head[0];
     i_x <<= 8;
     i_x |= i_head[1];
     i_x <<= 8;
     i_x |= i_head[2];
     i_x <<= 8;
     i_x |= i_head[3];

     fread(&i_head, 1, 4, fi);

     i_y = i_head[0];
     i_y <<= 8;
     i_y |= i_head[1];
     i_y <<= 8;
     i_y |= i_head[2];
     i_y <<= 8;
     i_y |= i_head[3];

     if (i_n != X->size1 || l_n != Y->size1 || i_x*i_y != X->size2) {
	  fprintf(stderr, "Malformed data in files: %s, %s\n", images, labels);
	  exit(EXIT_FAILURE);
     }

     unsigned char i_curs;
     unsigned char l_curs;

     int i, j;
     for (i = 0; i < X->size1; i++) {
	  for (j = 0; j < X->size2; j++) {
	       fread(&i_curs, 1, 1, fi);
	       gsl_matrix_set(X, i, j, i_curs / 255.0);
	  }

	  fread(&l_curs, 1, 1, fl);

	  if (l_curs > 9) {
	       fprintf(stderr, "Illegal label for image %d\n", i+1);
	       exit(EXIT_FAILURE);
	  }

	  gsl_matrix_set(Y, i, l_curs, 1.0);
     }

     fclose(fi);
     fclose(fl);
}

#endif
