import cv2
import numpy as np
from tkinter import *
from os import listdir
from os.path import isfile, join
import scipy
import threading
import pydicom
import datetime
from pydicom.dataset import Dataset, FileDataset
from pathlib import Path

import pydicom._storage_sopclass_uids

mean_color_on_img = 1


def input_data(img):
    win = Toplevel()

    win.resizable(False, False)
    win.geometry("400x400")
    win.title("Dane")

    name_label = Label(win, text="Imie", font='serif').place(x=20, y=50)
    surname_label = Label(win, text="Nazwisko", font='serif').place(x=20, y=100)
    id_label = Label(win, text="ID", font='serif').place(x=20, y=150)
    date_label = Label(win, text="Data", font='serif').place(x=20, y=200)
    comment_label = Label(win, text="Komentarz", font='serif').place(x=20, y=250)
    filename_label = Label(win, text="Nazwa pliku", font='serif').place(x=20, y=300)
    name_entry = Entry(win, bd=5)
    name_entry.insert(0, "Mateusz")
    name_entry.place(x=150, y=50)
    surname_entry = Entry(win, bd=5)
    surname_entry.insert(0, "Zelazowski")
    surname_entry.place(x=150, y=100)
    id_entry = Entry(win, bd=5)
    id_entry.insert(0, "10")
    id_entry.place(x=150, y=150)
    date_entry = Entry(win, bd=5)
    date_entry.insert(0, "30.03.2021")
    date_entry.place(x=150, y=200)
    comment_entry = Entry(win, bd=5)
    comment_entry.insert(0, "Standardowy komentarz")
    comment_entry.place(x=150, y=250)
    filename_entry = Entry(win, bd=5)
    filename_entry.insert(0, "out")
    filename_entry.place(x=150, y=300)
    save = Button(win, command=lambda: threading.Thread(
        target=lambda: save_dicom(img, name_entry.get(), surname_entry.get(), id_entry.get(), date_entry.get(),
                                  comment_entry.get(), filename_entry.get(), win)
        ).start(), text="Save", height=1, width=25).place(x=200, y=350)
    close = Button(win, command=lambda: threading.Thread(target=lambda: win.destroy()
                                                         ).start(), text="Close", height=1, width=25).place(x=10, y=350)


def save_dicom(img, name, surname, id, date, comment, filename, win):
    win.destroy()
    img = img * 255
    img = img.astype(np.uint16)
    if filename[-4:] == ".dcm":
        full_filename = filename
    elif filename == "":
        full_filename = "name.dcm"
    else:
        sufix = ".dcm"
        full_filename = filename + sufix

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    if name == "" and surname == "":
        full_name = "name^surname"
    else:
        full_name = name + "^" + surname
    ds.PatientName = full_name
    if id == "":
        id = "1"
    ds.PatientID = id
    if comment != "":
        ds.ImageComments = comment
    if date != "":
        ds.ContentDate = date
    else:
        dt = datetime.datetime.now()
        ds.ContentDate = dt.strftime('%Y%m%d')

    ds.SOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.Columns = img.shape[1]
    ds.Rows = img.shape[0]
    ds.ImagePositionPatient = r"0\0\1"
    ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.PixelSpacing = r"1\1"
    ds.Modality = "CT"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
    ds.PixelData = img.tobytes()
    Path("output_dicom").mkdir(parents=True, exist_ok=True)
    ds.save_as('output_dicom/' + full_filename, write_like_original=False)


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
            points.append([i, j])
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
            points.append([i, j])
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
            points.append([i, j])
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
            points.append([i, j])
            if e >= 0:
                j += 1
                e += dx
            i += 1
            e += dy

    if change:
        points = [[y, x] for x, y in points]

    return points


def calculate_positions(alfa, phi, n, radius):
    positions = []
    for i in range(n):
        deg1 = alfa + phi / 2 - i * phi / (n - 1)
        x1 = radius * np.cos(deg1) + radius - 1
        y1 = radius * np.sin(deg1) + radius - 1
        deg = alfa + np.pi - phi / 2 + i * phi / (n - 1)
        xd = radius * np.cos(deg) + radius - 1
        yd = radius * np.sin(deg) + radius - 1
        positions.append([x1, y1, xd, yd])

    return positions


def draw_sinogram(l, n, angle, radius, img, deg, with_steps):
    angles = np.arange(0, deg, angle)
    sinogram = np.zeros((int(round(deg * (1 / angle))), n))
    ratio = 1 / angle
    lines = []

    divider = (15 * mean_color_on_img ** 0.7)
    for angle in angles:
        angle_deg = angle
        angle = np.deg2rad(angle)
        phi = np.deg2rad(l)
        sin_row = []
        lines_row = []
        positions = calculate_positions(angle, phi, n, radius)
        for position in positions:
            if int(position[0]) == int(position[2]) and int(position[1]) == int(position[3]):
                line = [[int(position[0]), int(position[1])]]
            else:
                line = bresenham2(int(position[0]), int(position[1]), int(position[2]), int(position[3]))
            line = np.array(line)
            color = sum(img[line[:, 0], line[:, 1]]) / len(line) / divider
            sin_row.append(color)
            lines_row.append(line)
        lines.append(lines_row)
        sinogram[int(angle_deg * ratio), :] = sin_row
        if with_steps:
            winname = "sinogram"
            cv2.imshow(winname, sinogram)
            cv2.waitKey(1)
    return np.array(sinogram), lines


def reverse(l, sinogram, n, angle2, radius, lines, deg, with_steps):
    img = np.zeros((radius * 2, radius * 2))
    angles = np.arange(0, deg, angle2)
    ratio = 1 / angle2
    divider = n / 40 / np.mean(sinogram) * ratio * (180 / l) ** 2
    for angle in angles:
        i = 0
        for line in lines[int(angle * ratio)]:
            color = sinogram[int(angle * ratio)][i]
            i += 1
            img[line[:, 0], line[:, 1]] += color / divider
        if with_steps:
            winname = "reverse"
            cv2.imshow(winname, img)
            cv2.waitKey(1)

    return img


def reverse_filtered(l, sinogram, n, angle2, radius, lines, deg, with_steps):
    img = np.zeros((radius * 2, radius * 2))
    angles = np.arange(0, deg, angle2)
    ratio = 1 / angle2
    divider = n / 750 / np.mean(sinogram) ** 1.8 * ratio / 2.5 * (180 / l) ** 2
    for i in range(len(sinogram)):
        sinogram[i] = np.real(scipy.fft.ifft(scipy.fft.fft((2 * sinogram[i]) ** 1.25 - 0.3)
                                             * np.hamming(len(sinogram[i]))))

    for angle in angles:
        i = 0
        for line in lines[int(angle * ratio)]:
            color = sinogram[int(angle * ratio)][i]
            i += 1
            img[line[:, 0], line[:, 1]] += color / divider
        if with_steps:
            winname = "reverse-filtered"
            cv2.imshow(winname, img)
            cv2.waitKey(1)

    return img


def show(l, n, angle, filtr, both, with_steps):
    cv2.destroyAllWindows()
    global mean_color_on_img
    file = images.get()

    processing_label = Label(app, text="Przetwarzanie obrazu ... ", font=('serif', 9))
    processing_label.place(x=120, y=332)

    if file[-4:] == ".dcm":
        img1 = pydicom.dcmread('tomograf-zdjecia/' + file)
        s = max(img1.pixel_array.shape)
        img2 = np.zeros((s, s), np.uint8)
        ax, ay = (s - img1.pixel_array.shape[1]) // 2, (s - img1.pixel_array.shape[0]) // 2
        img2[ay:img1.pixel_array.shape[0] + ay, ax:ax + img1.pixel_array.shape[1]] = img1.pixel_array
    else:
        img1 = cv2.imread('tomograf-zdjecia/' + file, 0).astype('float64')
        s = max(img1.shape[0:2])
        img2 = np.zeros((s, s), np.uint8)
        ax, ay = (s - img1.shape[1]) // 2, (s - img1.shape[0]) // 2
        img2[ay:img1.shape[0] + ay, ax:ax + img1.shape[1]] = img1
    img = img2
    radius = int(s / 2)
    mean_color_on_img = np.array(img).mean()
    deg = 180

    rev_filter = None
    rev = None
    sino, lines = draw_sinogram(l, n, angle, radius, img, deg, with_steps)
    if both:
        rev = reverse(l, sino, n, angle, radius, lines, deg, with_steps)
        rev_filter = reverse_filtered(l, sino, n, angle, radius, lines, deg, with_steps)
    elif filtr:
        rev_filter = reverse_filtered(l, sino, n, angle, radius, lines, deg, with_steps)
    else:
        rev = reverse(l, sino, n, angle, radius, lines, deg, with_steps)

    if not with_steps:
        winname = "sinogram"
        cv2.imshow(winname, sino)
        cv2.waitKey(1)

        if rev is not None:
            winname = "Przetworzony obraz"
            cv2.imshow(winname, rev)
            cv2.waitKey(1)

        if rev_filter is not None:
            winname = "Przetworzony obraz z filtrowaniem"
            cv2.imshow(winname, rev_filter)
            cv2.waitKey(1)

    cv2.imshow("Obraz oryginalny", img)
    cv2.waitKey(1)

    processing_label.destroy()

    if rev_filter is not None:
        rev_filter[rev_filter < 0] = 0
        rev_filter[rev_filter > 1] = 1
        MSE = 0
        for i in range(len(rev_filter)):
            for j in range(len(rev_filter[i])):
                MSE += (img[i][j] - rev_filter[i][j] * 255) ** 2
        MSE = MSE / (len(rev_filter) * len(rev_filter[0]))
        RMSE = MSE ** 0.5
        print("RMSE filtered =", RMSE)

    if rev is not None:
        rev[rev < 0] = 0
        rev[rev > 1] = 1
        MSE = 0
        for i in range(len(rev)):
            for j in range(len(rev[i])):
                MSE += (img[i][j] - rev[i][j] * 255) ** 2
        MSE = MSE / (len(rev) * len(rev[0]))
        RMSE = MSE ** 0.5
        print("RMSE =", RMSE)
    if rev_filter is not None:
        input_data(rev_filter)
    else:
        input_data(rev)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app = Tk()
    app.resizable(False, False)
    app.geometry("400x460")
    app.title("Tomograf")
    name = Label(app, text="Tomograf", font=('serif', 20)).place(x=130, y=20)
    images = StringVar(app)
    files = [f for f in listdir("tomograf-zdjecia") if isfile(join("tomograf-zdjecia", f))]
    images_names = files
    images.set(images_names[0])
    images_list = OptionMenu(app, images, *images_names)
    images_list.config(width=27)
    images_list.place(x=100, y=100)
    img_label = Label(app, text="Obraz:", font='serif').place(x=20, y=105)
    l_label = Label(app, text="rozpiętość (l):", font='serif').place(x=20, y=150)
    n_label = Label(app, text="liczba detektorów (n):", font='serif').place(x=20, y=200)
    angle_label = Label(app, text="Krok ∆α:", font='serif').place(x=20, y=250)
    l_entry = Entry(app, bd=5)
    l_entry.insert(0, "180")
    l_entry.place(x=225, y=150)
    n_entry = Entry(app, bd=5)
    n_entry.insert(0, "150")
    n_entry.place(x=225, y=200)
    angle_entry = Entry(app, bd=5)
    angle_entry.insert(0, "1")
    angle_entry.place(x=225, y=250)
    filtr = IntVar()
    both = IntVar()
    with_steps = IntVar()
    c1 = Checkbutton(app, text='Pokaż obraz z filtrem', variable=filtr, onvalue=1, offvalue=0)
    c1.pack()
    c1.place(x=50, y=360)
    c2 = Checkbutton(app, text='Pokaż oba obrazy', variable=both, onvalue=1, offvalue=0)
    c2.pack()
    c2.place(x=50, y=390)
    c3 = Checkbutton(app, text='Z pokazywaniem kroków', variable=with_steps, onvalue=1, offvalue=0)
    c3.pack()
    c3.place(x=50, y=420)
    start = Button(app, command=lambda: threading.Thread(
        target=lambda: show(int(l_entry.get()), int(n_entry.get()), float(angle_entry.get()), filtr.get(), both.get(),
                            with_steps.get())).start(),
                   text="Show results", height=1, width=28).place(x=95, y=300)
    app.mainloop()
