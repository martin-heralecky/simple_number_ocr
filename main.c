#include <stdio.h>
#include <string.h>

#define IMAGE_WIDTH         28
#define IMAGE_HEIGHT        28
#define TRAIN_IMAGES_NUM 60000
#define TEST_IMAGES_NUM  10000

#define MAX_NEURON_VALUE 2147483647u

unsigned int map(unsigned int from_max, unsigned int to_max, unsigned int val)
{
    return (unsigned long long)val * to_max / from_max;
}

void load_image(FILE *fd, unsigned char *image)
{
    int i;
    for (i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i) {
        image[i] = (unsigned char)fgetc(fd);
    }
}

void load_label(FILE *fd, unsigned char *label)
{
    *label = (unsigned char)fgetc(fd);
}

void print_image(unsigned char *image)
{
    int row, col;
    for (row = 0; row < IMAGE_HEIGHT; ++row) {
        for (col = 0; col < IMAGE_WIDTH; ++col) {
            printf("%3u ", image[row * IMAGE_WIDTH + col]);
        }

        putc('\n', stdout);
    }
}

void print_net(unsigned int *net)
{
    int row, col;
    for (row = 0; row < IMAGE_HEIGHT; ++row) {
        for (col = 0; col < IMAGE_WIDTH; ++col) {
            printf(
                "%3u ",
                map(MAX_NEURON_VALUE, 255, net[row * IMAGE_WIDTH + col])
            );
        }

        putc('\n', stdout);
    }
}

void print_label(unsigned char label)
{
    printf("%u\n", label);
}

void learn(unsigned int *net, const unsigned char *image, int *learned_num)
{
    int i;

    for (i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i) {
        net[i] =
            (
                (unsigned long long)net[i] *
                *learned_num +
                map(255, MAX_NEURON_VALUE, image[i])
            ) / (*learned_num + 1);
    }

    ++(*learned_num);
}

unsigned char guess(
    unsigned int net[][IMAGE_WIDTH * IMAGE_HEIGHT],
    const unsigned char *image)
{
    int i, row, col, guess = -1;
    long double val, highest_val = -1000000;

    for (i = 0; i < 10; ++i) {
        val = 0;

        for (row = 0; row < IMAGE_HEIGHT; ++row) {
            for (col = 0; col < IMAGE_WIDTH; ++col) {
                val += image[row * IMAGE_WIDTH + col] *
                    (
                        (long double)(net[i][row * IMAGE_WIDTH + col]) /
                        MAX_NEURON_VALUE -
                        0.5
                    );
            }
        }

        if (val > highest_val) {
            highest_val = val;
            guess = i;
        }
    }

    return guess;
}

int main(void)
{
    unsigned int net[10][IMAGE_WIDTH * IMAGE_HEIGHT];
    unsigned char image[IMAGE_WIDTH * IMAGE_HEIGHT],
        label, guess_label;
    int learned_nums[10] = {0}, i, correct_guesses = 0;

    memset(net, 0, 10 * IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(unsigned int));

    FILE *train_images_fd,
        *train_labels_fd,
        *test_images_fd,
        *test_labels_fd;

    train_images_fd = fopen("train-images-idx3-ubyte", "rb");
    fseek(train_images_fd, 16, SEEK_SET);

    train_labels_fd = fopen("train-labels-idx1-ubyte", "rb");
    fseek(train_labels_fd, 8, SEEK_SET);

    printf("Learning...\n");

    for (i = 0; i < TRAIN_IMAGES_NUM; ++i) {
        load_image(train_images_fd, image);
        load_label(train_labels_fd, &label);

        learn(net[label], image, &learned_nums[label]);
    }

    printf("Done.\nPrinting net:\n\n");

    for (i = 0; i < 10; ++i) {
        print_net(net[i]);
        print_label(i);
    }

    printf("\nTesting...\n");

    test_images_fd = fopen("t10k-images-idx3-ubyte", "rb");
    fseek(test_images_fd, 16, SEEK_SET);

    test_labels_fd = fopen("t10k-labels-idx1-ubyte", "rb");
    fseek(test_labels_fd, 8, SEEK_SET);

    for (i = 0; i < TEST_IMAGES_NUM; ++i) {
        load_image(test_images_fd, image);
        load_label(test_labels_fd, &label);

        guess_label = guess(net, image);

        printf("Guessed label: %c   Actual label: %u", guess_label == 255 ? 'N' : '0' + guess_label, label);
        if (guess_label != label) {
            printf("   X");
        } else {
            ++correct_guesses;
        }
        putc('\n', stdout);
    }

    printf("Success rate: %.2lf%%\n", (double)correct_guesses / TEST_IMAGES_NUM * 100);

    return 0;
}
