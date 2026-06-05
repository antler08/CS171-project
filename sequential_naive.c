//sequential

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <gmp.h>

static void mod_exp_naive(mpz_t result,
                          const mpz_t base,
                          const mpz_t exp,
                          const mpz_t mod)
{
    mpz_t b, tmp;
    mpz_inits(b, tmp, NULL);

    mpz_mod(b, base, mod);          
    mpz_set_ui(result, 1);          

    size_t nbits = mpz_sizeinbase(exp, 2);

    for (long i = (long)nbits - 1; i >= 0; i--) {

        //square 
        mpz_mul(tmp, result, result);
        mpz_mod(result, tmp, mod);

        //multiply step
        if (mpz_tstbit(exp, (mp_bitcnt_t)i)) {
            mpz_mul(tmp, result, b);
            mpz_mod(result, tmp, mod);
        }
    }

    mpz_clears(b, tmp, NULL);
}


int main(int argc, char *argv[]) {

    //for checking
    {
        mpz_t b, e, m, r;
        mpz_inits(b, e, m, r, NULL);

        mpz_set_ui(b, 3);
        mpz_set_ui(e, 11);
        mpz_set_ui(m, 17);

        mod_exp_naive(r, b, e, m);

        unsigned long val = mpz_get_ui(r);
        printf("3^11 mod 17 = %lu (expected 7)\n", val);

        if (val != 7) {
            fprintf(stderr, "arithmethic fail\n");
            mpz_clears(b, e, m, r, NULL);
            return 1;
        }

        mpz_clears(b, e, m, r, NULL);
    }

    //arg based input
    if (argc < 2) {
        fprintf(stderr,
            "    %s input_file [output_file]\n\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    const char *inputFile  = argv[1];
    const char *outputFile = (argc >= 3) ? argv[2]
                                         : "../data/out/sequential_naive_results.txt";


    //arg error
    FILE *fin = fopen(inputFile, "r");
    if (!fin) {
        perror("Error opening input file");
        return 1;
    }

    FILE *fout = fopen(outputFile, "w");
    if (!fout) {
        perror("Error opening output file");
        fclose(fin);
        return 1;
    }

    mpz_t base, exp, mod, result;
    mpz_inits(base, exp, mod, result, NULL);

    char line[4096];
    int  count   = 0;
    int  skipped = 0;

    struct timespec ts, te;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    while (fgets(line, sizeof(line), fin)) {

        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
            continue;

        line[strcspn(line, "\r\n")] = '\0';

        char *tok_b = strtok(line,  " \t");
        char *tok_e = strtok(NULL,  " \t");
        char *tok_m = strtok(NULL,  " \t");

        if (!tok_b || !tok_e || !tok_m) {
            fprintf(stderr, "Skipping malformed line (need 3 tokens)\n");
            skipped++;
            continue;
        }

        int ok = 1;
        ok &= (mpz_set_str(base, tok_b, 0) == 0);
        ok &= (mpz_set_str(exp,  tok_e, 0) == 0);
        ok &= (mpz_set_str(mod,  tok_m, 0) == 0);

        if (!ok) {
            fprintf(stderr, "Skipping unparseable line\n");
            skipped++;
            continue;
        }

        if (mpz_sgn(mod) == 0) {
            fprintf(stderr, "Skipping modulus = 0\n");
            skipped++;
            continue;
        }

        mod_exp_naive(result, base, exp, mod);

        char *res_str = mpz_get_str(NULL, 10, result);
        fprintf(fout, "%s\n", res_str);
        free(res_str);

        count++;
    }

    //benchmarking
    clock_gettime(CLOCK_MONOTONIC, &te);

    double ms = (te.tv_sec  - ts.tv_sec)  * 1000.0
              + (te.tv_nsec - ts.tv_nsec) / 1e6;

    printf("Input  file : %s\n",  inputFile);
    printf("Output file : %s\n",  outputFile);
    printf("Processed   : %d cases\n", count);
    if (skipped)
        printf("Skipped     : %d lines\n", skipped);
    printf("Total time  : %.4f ms\n", ms);

    mpz_clears(base, exp, mod, result, NULL);
    fclose(fin);
    fclose(fout);

    return 0;
}