import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from os import listdir
from os.path import isfile, join

mean_color_on_img = 1


def bresenham2(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    change = False
    if abs(dx) < abs(dy):
        change = True
        x1, y1, x2, y2 = y1, x1, y2, x2
        dx = x2 - x1
        dy = y2 - y1

    if dx >= 0 and dy >= 0:
        j = y1
        e = dy - dx
        points = []

        for i in range(x1, x2):
            points.append((i, j))
            if e >= 0:
                j += 1
                e -= dx
            i += 1
            e += dy

    elif dx >= 0 and dy < 0:
        j = y1
        e = -dy - dx
        points = []

        for i in range(x1, x2):
            points.append((i, j))
            if e >= 0:
                j -= 1
                e -= dx
            i += 1
            e -= dy

    elif dx < 0 and dy < 0:
        j = y1
        e = -dy + dx
        points = []

        for i in range(x1, x2, -1):
            points.append((i, j))
            if e >= 0:
                j -= 1
                e += dx
            i += 1
            e -= dy

    else:
        j = y1
        e = dy + dx
        points = []

        for i in range(x1, x2, -1):
            points.append((i, j))
            if e >= 0:
                j += 1
                e += dx
            i += 1
            e += dy

    if change:
        points = [(y, x) for x, y in points]

    return points


def get_color(line, radius, img):
    result = []
    for x, y in line:
        result.append(img[x + radius - 1, y + radius - 1])
    return np.array(result).mean() / (20 * mean_color_on_img ** 0.6)


def calculate_positions(alfa, phi, n, radius):
    positions = []
    x1 = radius * np.cos(alfa)
    y1 = radius * np.sin(alfa)
    for i in range(n):
        deg = alfa + np.pi - phi / 2 + i * phi / (n - 1)
        xd = radius * np.cos(deg)
        yd = radius * np.sin(deg)
        positions.append([x1, y1, xd, yd])

    return positions


def process(l, n, angle, radius, img, deg):
    angles = np.arange(0, deg, angle)
    sinogram = np.zeros((int(deg * (1 / angle)), n))
    ratio = 1 / angle
    lines = []
    for angle in angles:
        angle_deg = angle
        angle = np.deg2rad(angle)
        phi = np.deg2rad(l)
        sin_row = []
        lines_row = []
        positions = calculate_positions(angle, phi, n, radius)
        for position in positions:
            line = bresenham2(int(position[0]), int(position[1]), int(position[2]), int(position[3]))
            color = get_color(line, radius, img)
            sin_row.append(color)
            lines_row.append(line)
        # sinogram.append(sin_row)
        lines.append(lines_row)
        sinogram[int(angle_deg * ratio), :] = sin_row
        winname = "sinogram"
        cv2.namedWindow(winname)  # Create a named window
        cv2.moveWindow(winname, 650, 200)  # Move it to (40,30)
        cv2.imshow(winname, sinogram)
        cv2.waitKey(1)
    return np.array(sinogram), lines


def draw(img, line, color, radius, n):
    color = color / (2 * n / 3)
    for i in line:
        img[i[0] + radius - 1][i[1] + radius - 1] += color
    return img


def draw2(img, lines, colors, radius, n):
    # color = color/(2*n/3)
    c = 0
    for line in lines:
        for i in line:
            if radius * 2 > (i[0] + radius - 1) >= 0 and radius * 2 > (i[1] + radius - 1) >= 0:
                img[i[0] + radius - 1][i[1] + radius - 1] += colors[c] / (4 * n)
        c += 1
    return img


def reverse(sinogram, n, angle2, radius, lines, deg):
    img = np.zeros((radius * 2, radius * 2))
    angles = np.arange(0, deg, angle2)
    ratio = 1 / angle2
    for angle in angles:
        i = 0
        for line in lines[int(angle * ratio)]:
            color = sinogram[int(angle * ratio)][i]
            i += 1
            img = draw(img, line, color, radius, n)
        winname = "reverse"
        # cv2.namedWindow(winname)  # Create a named window
        # cv2.moveWindow(winname, 1000, 100)  # Move it to (40,30)
        cv2.imshow(winname, img)
        cv2.waitKey(1)

    return img


def reverse2(sinogram, n, angle2, radius, lines, deg):
    img2 = np.zeros((radius * 2, radius * 2))
    angles = np.arange(0, deg, angle2)
    ratio = 1 / angle2
    p = np.random.permutation(len(angles))
    angles = angles[p]

    for angle in angles:
        ii = 0
        for line in lines[int(angle * ratio)]:
            lines2 = []

            for i in range(13):
                lin = []
                for c in line:
                    lin.append((c[0] - 2 + i, c[1] - 2 + i))
                lines2.append(lin)

            color = sinogram[int(angle * ratio)][ii]
            ii += 1
            filtered = [-0.1 + color, -0.2 + color, -0.3 + color, -0.4 + color, -0.6 + color, color, color, color, -0.6 + color,
                        -0.4 + color, -0.3 + color, -0.2 + color,-0.1 + color]
            img2 = draw2(img2, lines2, filtered, radius, n)

        winname = "reverse_filtered"
        # cv2.namedWindow(winname)  # Create a named window
        # cv2.moveWindow(winname, 1200, 100)  # Move it to (40,30)
        cv2.imshow(winname, img2)
        cv2.waitKey(1)

    return img2


def show(l, n, angle, filtr, both, full):
    global mean_color_on_img
    img1 = cv2.imread('tomograf-zdjecia/' + images.get())  # , 0).astype('float64')
    # width, height = np.shape(img1)
    s = max(img1.shape[0:2])
    img2 = np.zeros((s, s, 3), np.uint8)
    ax, ay = (s - img1.shape[1]) // 2, (s - img1.shape[0]) // 2
    img2[ay:img1.shape[0] + ay, ax:ax + img1.shape[1]] = img1
    radius = int(s / 2)
    img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mean_color_on_img = np.array(img).mean()
    if full:
        deg = 360
    else:
        deg = 180

    sino, lines = process(l, n, angle, radius, img, deg)
    if both:
        reverse(sino, n, angle, radius, lines, deg)
        reverse2(sino, n, angle, radius, lines, deg)
    elif filtr:
        reverse2(sino, n, angle, radius, lines, deg)
    else:
        reverse(sino, n, angle, radius, lines, deg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app = Tk()
    app.resizable(False, False)
    app.geometry("400x400")
    app.title("Tomograf")
    name = Label(app, text="Tomograf", font=('serif', 20)).place(x=130, y=50)
    images = StringVar(app)
    files = [f for f in listdir("tomograf-zdjecia") if isfile(join("tomograf-zdjecia", f))]
    images_names = files
    images.set(images_names[0])
    images_list = OptionMenu(app, images, *images_names)
    images_list.config(width=27)
    images_list.place(x=100, y=100)
    l_label = Label(app, text="l", font='serif').place(x=100, y=150)
    n_label = Label(app, text="n", font='serif').place(x=100, y=200)
    angle_label = Label(app, text="angle", font='serif').place(x=80, y=250)
    l_entry = Entry(app, bd=5)
    l_entry.insert(0, "270")
    l_entry.place(x=150, y=150)
    n_entry = Entry(app, bd=5)
    n_entry.insert(0, "150")
    n_entry.place(x=150, y=200)
    angle_entry = Entry(app, bd=5)
    angle_entry.insert(0, "1")
    angle_entry.place(x=150, y=250)
    filtr = IntVar()
    both = IntVar()
    full = IntVar()
    c1 = Checkbutton(app, text='Filtr', variable=filtr, onvalue=1, offvalue=0)
    c1.pack()
    c1.place(x=100, y=350)
    c2 = Checkbutton(app, text='Oba', variable=both, onvalue=1, offvalue=0)
    c2.pack()
    c2.place(x=180, y=350)
    c3 = Checkbutton(app, text='360°', variable=full, onvalue=1, offvalue=0)
    c3.pack()
    c3.place(x=260, y=350)
    a = Button(app, command=lambda: show(int(l_entry.get()), int(n_entry.get()), float(angle_entry.get()), filtr.get(), both.get(), full.get()),
               text="Show results", height=1, width=28).place(x=100, y=300)

    app.mainloop()
