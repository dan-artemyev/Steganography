import math
from PIL import Image

bits = 8


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


def main():
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
        for i in range(8):
            for j in range(8):
                count = 0
                if length <= leng:
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
        if length >= leng:
            break
        width_block += 1
        if width_block == width_blocks:
            height_block += 1
            width_block = 0
    result = get_secret(buf)
    print(get_secret(buf))


if __name__ == '__main__':
    main()