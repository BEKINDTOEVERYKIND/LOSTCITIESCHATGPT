/* symmetrize -- remove arbitrary physical-ID distinctions between wagers. */
#include "../src/net.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s INPUT.bin OUTPUT.bin\n", argv[0]);
        return 1;
    }
    Net *net = (Net *)malloc(sizeof(*net));
    if (!net) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    if (net_load(net, argv[1]) != 0) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        free(net);
        return 1;
    }
    net_project_wager_symmetry(net);
    if (net_save(net, argv[2]) != 0) {
        fprintf(stderr, "cannot save %s\n", argv[2]);
        free(net);
        return 1;
    }
    free(net);
    printf("wrote wager-symmetric model to %s\n", argv[2]);
    return 0;
}
