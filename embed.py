import math
from PIL import Image

bits = 8
source = "example2.bmp"
source_secret = "source_secret.bmp"


def get_binary(secret):
    result = []
    # secret = str(len(secret)) + ":" + secret
    for letter in secret:
        letter_bin = list(bin(ord(letter))[2:])
        while len(letter_bin) < bits:
            letter_bin.insert(0, '0')
        result += letter_bin
    return result


def my_round(x):
    if x < 0:
        if x % 1 > 0.5:
            x = int(x)
        elif x % 1 <= 0.5:
            x = int(x) - 1

    elif x >= 0:
        if x % 1 >= 0.5:
            x = int(x) + 1
        elif x % 1 < 0.5:
            x = int(x)
    return x


def main():
    print("Enter your message:")
    secret = input()
    secret = get_binary(secret)
    img = Image.open(source)
    width, height = img.size
    img.close()

    red = [width * [0] for i in range(height)]
    green = [width * [0] for i in range(height)]
    blue = [width * [0] for i in range(height)]

    with open(source, 'rb') as img:
        img_bytes = bytearray(img.read())
        i = 54
        x_red = 0
        x_green = 0
        x_blue = 0
        y_red = 0
        y_green = 0
        y_blue = 0

        while x_red != height:
            if (i - 53) % 3 == 0 and i < len(img_bytes):
                red[x_red][y_red] = img_bytes[i]
                y_red += 1
            elif (i - 53) % 3 == 1 and i < len(img_bytes):
                blue[x_blue][y_blue] = img_bytes[i]
                y_blue += 1
            elif (i - 53) % 3 == 2 and i < len(img_bytes):
                green[x_green][y_green] = img_bytes[i]
                y_green += 1
            i += 1
            if y_red == width:
                y_red = 0
                x_red += 1
            if y_green == width:
                y_green = 0
                x_green += 1
            if y_blue == width:
                y_blue = 0
                x_blue += 1

    red8 = 8 * [0]
    green8 = 8 * [0]
    blue8 = 8 * [0]
    dct_red = 8 * [0]
    dct_green = 8 * [0]
    dct_blue = 8 * [0]

    for i in range (8):
        red8[i] = 8 * [0]
        green8[i] = 8 * [0]
        blue8[i] = 8 * [0]
        dct_red[i] = 8 * [0]
        dct_green[i] = 8 * [0]
        dct_blue[i] = 8 * [0]

    height_block_number = 0
    width_block_number = 0
    height_blocks = int(height / 8) - 1
    width_blocks = int(width / 8)
    length = 0

    # making 8x8 blocks
    while height_block_number <= height_blocks and length < len(secret):
        for i in range(8):
            for j in range(8):
                red8[i][j] = red[height_block_number * 8 + i][width_block_number * 8 + j]
                green8[i][j] = green[height_block_number * 8 + i][width_block_number * 8 + j]
                blue8[i][j] = blue[height_block_number * 8 + i][width_block_number * 8 + j]
        # DCT
        for u in range(8):
            for v in range(8):
                if u == 0:
                    cu = math.sqrt(1 / 2)
                else:
                    cu = 1
                if v == 0:
                    cv = math.sqrt(1 / 2)
                else:
                    cv = 1
                sum_red = 0
                sum_green = 0
                sum_blue = 0
                for x in range(8):
                    for y in range(8):
                        sum_red += red8[x][y] * math.cos(math.pi * (2 * y + 1) * v / 16) * math.cos(
                            math.pi * (2 * x + 1) * u / 16)
                        sum_green += green8[x][y] * math.cos(math.pi * (2 * y + 1) * v / 16) * math.cos(
                            math.pi * (2 * x + 1) * u / 16)
                        sum_blue += blue8[x][y] * math.cos(math.pi * (2 * y + 1) * v / 16) * math.cos(
                            math.pi * (2 * x + 1) * u / 16)
                dct_red[u][v] = (sum_red * cu * cv) / (math.sqrt(16))
                dct_green[u][v] = (sum_green * cu * cv) / (math.sqrt(16))
                dct_blue[u][v] = (sum_blue * cu * cv) / (math.sqrt(16))

        # Embedding
        for i in range(8):
            for j in range(8):
                if length < len(secret):
                    dct_red[i][j] = int(dct_red[i][j])
                    dct_green[i][j] = int(dct_green[i][j])
                    dct_blue[i][j] = int(dct_blue[i][j])

                    if secret[length] == '1':
                        if int(dct_red[i][j]) % 2 != 1:
                            dct_red[i][j] = int(dct_red[i][j]) + 1
                        if int(dct_green[i][j]) % 2 != 1:
                            dct_green[i][j] = int(dct_green[i][j]) + 1
                        if int(dct_blue[i][j]) % 2 != 1:
                            dct_blue[i][j] = int(dct_blue[i][j]) + 1
                    elif secret[length] == '0':
                        if int(dct_red[i][j]) % 2 != 0:
                            dct_red[i][j] = int(dct_red[i][j]) - 1
                        if int(dct_green[i][j]) % 2 != 0:
                            dct_green[i][j] = int(dct_green[i][j]) - 1
                        if int(dct_blue[i][j]) % 2 != 0:
                            dct_blue[i][j] = int(dct_blue[i][j]) - 1
                    length += 1
                else:
                    dct_red[i][j] = my_round(dct_red[i][j])
                    dct_green[i][j] = my_round(dct_green[i][j])
                    dct_blue[i][j] = my_round(dct_blue[i][j])

        # DCT back
        for x in range(8):
            for y in range(8):
                sum_red = 0
                sum_green = 0
                sum_blue = 0
                for u in range(8):
                    for v in range(8):
                        if u == 0:
                            cu = math.sqrt(1 / 2)
                        else:
                            cu = 1
                        if v == 0:
                            cv = math.sqrt(1 / 2)
                        else:
                            cv = 1
                        sum_red += cu * cv * dct_red[u][v] * math.cos(math.pi * (2 * y + 1) * v / 16) * math.cos(
                            math.pi * (2 * x + 1) * u / 16)
                        sum_green += cu * cv * dct_green[u][v] * math.cos(math.pi * (2 * y + 1) * v / 16) * math.cos(
                            math.pi * (2 * x + 1) * u / 16)
                        sum_blue += cu * cv * dct_blue[u][v] * math.cos(math.pi * (2 * y + 1) * v / 16) * math.cos(
                            math.pi * (2 * x + 1) * u / 16)
                red8[x][y] = sum_red / (math.sqrt(16))
                green8[x][y] = sum_green / (math.sqrt(16))
                blue8[x][y] = sum_blue / (math.sqrt(16))

                red8[x][y] = my_round(red8[x][y])
                green8[x][y] = my_round(green8[x][y])
                blue8[x][y] = my_round(blue8[x][y])

                if red8[x][y] > 255:
                    red8[x][y] = 255
                elif red8[x][y] < 0:
                    red8[x][y] = 0

                if green8[x][y] > 255:
                    green8[x][y] = 255
                elif green8[x][y] < 0:
                    green8[x][y] = 0

                if blue8[x][y] > 255:
                    blue8[x][y] = 255
                elif blue8[x][y] < 0:
                    blue8[x][y] = 0

                red[height_block_number * 8 + x][width_block_number * 8 + y] = red8[x][y]
                green[height_block_number * 8 + x][width_block_number * 8 + y] = green8[x][y]
                blue[height_block_number * 8 + x][width_block_number * 8 + y] = blue8[x][y]
        width_block_number += 1
        if width_block_number == width_blocks:
            height_block_number += 1
            width_block_number = 0

    # Building image
    i = 54
    x_red = 0
    x_green = 0
    x_blue = 0
    y_red = 0
    y_green = 0
    y_blue = 0
    while x_red != height:
        if (i - 53) % 3 == 0 and i < len(img_bytes):
            img_bytes[i] = red[x_red][y_red]
            y_red += 1
        elif (i - 53) % 3 == 1 and i < len(img_bytes):
            img_bytes[i] = blue[x_blue][y_blue]
            y_blue += 1
        elif (i - 53) % 3 == 2 and i < len(img_bytes):
            img_bytes[i] = green[x_green][y_green]
            y_green += 1
        i += 1
        if y_red == width:
            y_red = 0
            x_red += 1
        if y_green == width:
            y_green = 0
            x_green += 1
        if y_blue == width:
            y_blue = 0
            x_blue += 1

    result = "result-" + str(len(secret)) + ".bmp"
    with open(result, 'wb') as image:
        image.write(img_bytes)


if __name__ == '__main__':
    main()