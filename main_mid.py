import math
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

bits = 8
source = "example2.bmp"
source_secret = "result.bmp"


def analyze(src, src_secret):
    max_value = 255.0
    img1 = cv2.imread(src)
    img2 = cv2.imread(src_secret)
    mse = np.mean((img1 - img2) ** 2)
    rmse = math.sqrt(mse)
    if mse != 0:
        psnr = 20 * math.log10(max_value / math.sqrt(mse))
        print('MSE = %.10f' % mse)
        print('RMSE = %.10f' % rmse)
        print('PSNR = %.2f' % psnr)
    elif mse == 0:
        print('MSE = 0\nPSNR = infinity')
    return float("%.2f" % psnr), rmse


def graph():
    secret = ''
    x = []
    y = []
    z = []
    for words in range(0, 50):
        secret += 'word '
        embed(secret)
        x.append(words)
        psnr, rmse = analyze(source, source_secret)
        y.append(psnr)
        z.append(rmse)

    fig, graph = plt.subplots()
    graph.plot(y, x)
    graph.grid()
    graph.set_xlabel('PSNR')
    graph.set_ylabel('Words amount')
    graph.set_title('Dependence of PSNR on words amount')
    fig, graph2 = plt.subplots()
    graph2.plot(z, x)
    graph2.grid()
    graph2.set_xlabel('RMSE')
    graph2.set_ylabel('Words amount')
    graph2.set_title('Dependence of RMSE on words amount')
    # plt.savefig('graph.png', bbox_inches='tight')
    plt.show()


def get_binary(secret):
    result = []
    # secret = str(len(secret)) + ":" + secret
    for letter in secret:
        letter_bin = list(bin(ord(letter))[2:])
        while len(letter_bin) < bits:
            letter_bin.insert(0, '0')
        result += letter_bin
    return result


def get_secret(data):
    result = ''
    data = "".join(data)
    for i in range(len(data) // bits):
        letter = chr(int(data[:bits], 2))
        data = data[bits:]
        result += letter
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


def embed(secret=''):
    if secret == '':
        print("Enter your message:")
        secret = input()
    print("Embedding message: " + secret)
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
        v = 1
        u = 7
        p = 0
        for i in range(8):
            for j in range(8):
                if j == u and v <= 2 and length < len(secret):
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
                    u += 1
                    v += 1
                    p += 1
                else:
                    dct_red[i][j] = my_round(dct_red[i][j])
                    dct_green[i][j] = my_round(dct_green[i][j])
                    dct_blue[i][j] = my_round(dct_blue[i][j])
                if v == 2:
                    v = 0
                    if p == 1:
                        u -= 2
                    else:
                        u -= 3

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

    # result = "result-" + str(len(secret)) + ".bmp"
    global source_secret
    source_secret = "result-" + str(len(secret)) + ".bmp"
    with open(source_secret, 'wb') as image:
        image.write(img_bytes)


def extract(filename=''):
    if filename == '':
        print('Enter filename:')
        filename = input()
    buf = []
    count = 1

    for i in range(len(filename)):
        if filename[i - count] == '-' and filename[i] != '.':
            count += 1
            buf += filename[i]
    stroka = ''.join(buf)
    leng = int(stroka)
    img = Image.open(filename)
    width, height = img.size
    img.close()

    red = [width * [0] for i in range(height)]
    green = [width * [0] for i in range(height)]
    blue = [width * [0] for i in range(height)]

    with open(filename, 'rb') as file:
        img_bytes = bytearray(file.read())
        i = 54
        x_red = 0
        x_blue = 0
        x_green = 0
        y_red = 0
        y_blue = 0
        y_green = 0
        while x_red != height:
            if (i - 53) % 3 == 1 and i < len(img_bytes):
                blue[x_blue][y_blue] = img_bytes[i]
                y_blue += 1
            elif (i - 53) % 3 == 2 and i < len(img_bytes):
                green[x_green][y_green] = img_bytes[i]
                y_green += 1
            elif (i - 53) % 3 == 0 and i < len(img_bytes):
                red[x_red][y_red] = img_bytes[i]
                y_red += 1
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

    for i in range(8):
        red8[i] = 8 * [0]
        green8[i] = 8 * [0]
        blue8[i] = 8 * [0]
        dct_red[i] = 8 * [0]
        dct_green[i] = 8 * [0]
        dct_blue[i] = 8 * [0]

    height_block = 0
    height_blocks = int(height / 8) - 1
    width_block = 0
    width_blocks = int(width / 8)
    buf = []
    length = 0
    while height_block <= height_blocks:
        for i in range(8):
            for j in range(8):
                red8[i][j] = red[height_block * 8 + i][width_block * 8 + j]
                green8[i][j] = green[height_block * 8 + i][width_block * 8 + j]
                blue8[i][j] = blue[height_block * 8 + i][width_block * 8 + j]
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
        # Extract
        v = 1
        u = 7
        p = 0
        for i in range(8):
            for j in range(8):
                count = 0
                if j == u and v <= 2 and length <= leng:
                    dct_red[i][j] = my_round(dct_red[i][j])
                    dct_green[i][j] = my_round(dct_green[i][j])
                    dct_blue[i][j] = my_round(dct_blue[i][j])

                    if dct_red[i][j] % 2 == 1:
                        count += 1
                    elif dct_red[i][j] % 2 == 0:
                        count += 0

                    if dct_green[i][j] % 2 == 1:
                        count += 1
                    elif dct_green[i][j] % 2 == 0:
                        count += 0

                    if dct_blue[i][j] % 2 == 1:
                        count += 1
                    elif dct_blue[i][j] % 2 == 0:
                        count += 0

                    if count >= 2:
                        buf += '1'
                    else:
                        buf += '0'

                    length += 1
                    u += 1
                    v += 1
                    p += 1

                if v == 2:
                    v = 0
                    if p == 1:
                        u -= 2
                    else:
                        u -= 3
        if length >= leng:
            break
        width_block += 1
        if width_block == width_blocks:
            height_block += 1
            width_block = 0
    result = get_secret(buf)
    print("Extracted message: " + result)


if __name__ == '__main__':
    embed('Hello, world!Hello, world!Hello, world!!')
    extract(source_secret)
    # extract("result-312.bmp")
    # analyze(source, source_secret)
    # graph()