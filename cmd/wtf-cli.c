/*
 * wtf-cli.c — CLI wrapper for WTForacle inference (libwtf.so)
 *
 * Usage: ./wtf-cli <weights.gguf> <prompt> [max_tokens] [temp]
 *
 * Build:
 *   go build -buildmode=c-shared -o libwtf.so ./wtf/
 *   cc -o wtf-cli cmd/wtf-cli.c -L. -lwtf -Wl,-rpath,'$ORIGIN'
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Go c-shared exports */
extern int wtf_init(char* weightsPath);
extern void wtf_free(void);
extern int wtf_generate(char* prompt, char* outputBuf, int maxOutputLen,
                        int maxTokens, float temperature, float topP,
                        char* anchorPrompt);

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <weights.gguf> <prompt> [max_tokens] [temp]\n", argv[0]);
        return 1;
    }

    const char* weights = argv[1];
    const char* prompt = argv[2];
    int max_tokens = argc > 3 ? atoi(argv[3]) : 150;
    float temp = argc > 4 ? (float)atof(argv[4]) : 0.9f;

    /* Redirect Go runtime prints to stderr */
    int rc = wtf_init((char*)weights);
    if (rc != 0) {
        fprintf(stderr, "[wtf-cli] init failed (rc=%d)\n", rc);
        return 1;
    }

    char buf[8192];
    memset(buf, 0, sizeof(buf));
    int n = wtf_generate((char*)prompt, buf, sizeof(buf), max_tokens, temp, 0.9f, NULL);

    /* Clean output to stdout */
    if (n > 0) {
        printf("%s\n", buf);
    }

    wtf_free();
    return 0;
}
