import os
import sys
from flask import Flask, render_template, Response, request, send_file
import cv2
import dlib
from scipy.spatial import distance as dist
import threading
from fpdf import FPDF
import io

def get_template_path():
    try:
        base_path = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    except Exception as e:
        base_path = os.path.dirname(os.path.abspath(__file__))
        print(f"Error in template path: {e}")
    return os.path.join(base_path, 'templates')

# Membuat aplikasi Flask
app = Flask(__name__, template_folder=get_template_path())

# Variabel global untuk menghitung kedipan, frame, deteksi mata tertutup, dan status deteksi
blink_count = 0
frame_count = 0
closed_eye_frames = 0
detection_result = ""
camera_active = True  # Status untuk menghentikan kamera

# Cek apakah sedang berjalan dalam environment PyInstaller
if getattr(sys, 'frozen', False):
    # Jika menggunakan PyInstaller, path file diambil dari sys.MEIPASS
    dat_path = os.path.join(sys._MEIPASS, "shape_predictor_68_face_landmarks.dat")
else:
    # Jika tidak, gunakan path file normal
    dat_path = "shape_predictor_68_face_landmarks.dat"

# Memuat model deteksi wajah dari dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dat_path)

# Indeks untuk mata kiri dan kanan dalam 68 poin wajah yang dideteksi
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# Fungsi untuk menghitung Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Jarak vertikal antara titik 1 dan 5
    B = dist.euclidean(eye[2], eye[4])  # Jarak vertikal antara titik 2 dan 4
    C = dist.euclidean(eye[0], eye[3])  # Jarak horizontal antara titik 0 dan 3
    ear = (A + B) / (2.0 * C)  # Rumus EAR
    return ear

# Fungsi untuk deteksi kedipan mata
def deteksi_kedipan():
    global blink_count, frame_count, closed_eye_frames, detection_result, camera_active
    ear_threshold = 0.25  # Batas EAR untuk deteksi mata tertutup
    consec_frames = 2  # Jumlah frame berturut-turut untuk mendeteksi kedipan

    # Buka kamera
    cap = cv2.VideoCapture(0)

    while camera_active:
        ret, frame = cap.read()  # Membaca frame dari kamera
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Ubah gambar ke grayscale
        faces = detector(gray, 0)  # Deteksi wajah pada frame

        for face in faces:
            shape = predictor(gray, face)  # Prediksi bentuk wajah menggunakan model
            shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]  # Mengambil 68 titik landmark wajah
            leftEye = shape[lStart:lEnd]  # Koordinat mata kiri
            rightEye = shape[rStart:rEnd]  # Koordinat mata kanan
            leftEAR = eye_aspect_ratio(leftEye)  # EAR mata kiri
            rightEAR = eye_aspect_ratio(rightEye)  # EAR mata kanan
            ear = (leftEAR + rightEAR) / 2.0  # EAR rata-rata kedua mata

            if ear < ear_threshold:  # Jika EAR di bawah ambang batas, hitung frame mata tertutup
                closed_eye_frames += 1
                frame_count += 1
            else:
                if frame_count >= consec_frames:  # Jika mata tertutup lebih dari jumlah frame yang ditentukan, hitung kedipan
                    blink_count += 1
                frame_count = 0  # Reset frame count jika mata terbuka

        # Logika untuk menentukan status deteksi
        if blink_count > 10 and closed_eye_frames <= 50:
            detection_result = "Normal"  # Kedipan cukup dan mata tidak lama tertutup
        elif closed_eye_frames > 100:
            detection_result = "Tunanetra"  # Mata tertutup terlalu lama
        elif blink_count == 0 and closed_eye_frames > 10:
            detection_result = "Tunanetra"  # Tidak ada kedipan dan mata tertutup lama
        else:
            detection_result = "Normal"  # Default ke normal

    cap.release()  # Menutup kamera
    cv2.destroyAllWindows()  # Menutup semua jendela OpenCV

# Route untuk halaman utama, menjalankan fungsi deteksi kedipan di thread terpisah
@app.route('/')
def index():
    global camera_active
    camera_active = True
    detection_thread = threading.Thread(target=deteksi_kedipan)  # Buat thread untuk deteksi
    detection_thread.daemon = True  # Buat thread menjadi daemon
    detection_thread.start()  # Mulai thread
    return render_template('form.html')  # Render halaman form

# Route untuk menerima dan memproses form
@app.route('/submit', methods=['POST'])
def submit_form():
    # Ambil data dari form
    nama = request.form['nama']
    email = request.form['email']
    tanggal_lahir = request.form['tanggal_lahir']
    alamat = request.form['alamat']
    pekerjaan = request.form['pekerjaan']
    hobi = request.form['hobi']
    jurusan = request.form['jurusan']
    jalur = request.form['jalur']
    nama_orang_tua = request.form['nama_orang_tua']
    alamat_orang_tua = request.form['alamat_orang_tua']
    phone_orang_tua = request.form['phone_orang_tua']
    pesan = request.form['pesan']
    
    global camera_active, detection_result
    camera_active = False  # Matikan kamera

    # Tampilkan hasil di halaman hasil
    return render_template('hasil.html', 
                           nama=nama, 
                           email=email, 
                           tanggal_lahir=tanggal_lahir, 
                           alamat=alamat, 
                           pekerjaan=pekerjaan, 
                           hobi=hobi, 
                           jurusan=jurusan,
                           jalur=jalur,
                           nama_orang_tua=nama_orang_tua,
                           alamat_orang_tua=alamat_orang_tua,
                           phone_orang_tua=phone_orang_tua,
                           pesan=pesan,
                           hasil_deteksi={detection_result})  # Menampilkan hasil deteksi


from fpdf import FPDF
import io
from flask import send_file

# Kelas PDF yang diperluas untuk menambahkan header dan footer
class PDF(FPDF):
    def header(self):
        # Logo (opsional)
        # self.image('path_to_logo.png', 10, 8, 33)
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'FORMULIR PENDAFTARAN MAHASISWA', ln=True, align='C')
        self.ln(2)  # Line break
        # Garis pertama
        self.set_line_width(0.5)  # Atur ketebalan garis
        self.line(10, self.get_y(), 200, self.get_y())  # Koordinat x1, y1, x2, y2
        # Line break
        self.ln(2)  # Jarak antara garis
        # Garis kedua
        self.line(10, self.get_y(), 200, self.get_y())  # Garis kedua
        self.ln(10)  # Line break setelah garis kedua
        

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        # Nomor halaman
        self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    # Ambil data dari form
    nama = request.form.get('nama', '')
    email = request.form.get('email', '')
    tanggal_lahir = request.form.get('tanggal_lahir', '')
    alamat = request.form.get('alamat', '')
    pekerjaan = request.form.get('pekerjaan', '')
    hobi = request.form.get('hobi', '')
    jurusan = request.form.get('jurusan', '')
    jalur = request.form.get('jalur', '')
    nama_orang_tua = request.form.get('nama_orang_tua', '')
    alamat_orang_tua = request.form.get('alamat_orang_tua', '')
    phone_orang_tua = request.form.get('phone_orang_tua', '')
    pesan = request.form.get('pesan', '')
    hasil_deteksi = request.form.get('hasil_deteksi', '')

    # Inisialisasi PDF
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Informasi Pribadi
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Informasi Pribadi', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.ln(2)  # Line break kecil

    pdf.cell(50, 10, 'Nama Lengkap', 0, 0)
    pdf.cell(0, 10, nama, ln=True)

    pdf.cell(50, 10, 'Email', 0, 0)
    pdf.cell(0, 10, email, ln=True)

    pdf.cell(50, 10, 'Tanggal Lahir', 0, 0)
    pdf.cell(0, 10, tanggal_lahir, ln=True)

    pdf.cell(50, 10, 'Alamat', 0, 0)
    pdf.multi_cell(0, 10, alamat)

    pdf.cell(50, 10, 'Pekerjaan', 0, 0)
    pdf.cell(0, 10, pekerjaan, ln=True)

    pdf.cell(50, 10, 'Hobi', 0, 0)
    pdf.cell(0, 10, hobi, ln=True)

    pdf.cell(50, 10, 'Jurusan', 0, 0)
    pdf.cell(0, 10, jurusan, ln=True)

    pdf.cell(50, 10, 'Jalur Pendaftaran', 0, 0)
    pdf.cell(0, 10, jalur, ln=True)

    # Garis Pembatas
    pdf.ln(5)
    pdf.set_draw_color(0, 0, 0)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)

    # Informasi Orang Tua
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Informasi Orang Tua', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.ln(2)

    pdf.cell(50, 10, 'Nama Orang Tua', 0, 0)
    pdf.cell(0, 10, nama_orang_tua, ln=True)

    pdf.cell(50, 10, 'Alamat Orang Tua', 0, 0)
    pdf.multi_cell(0, 10, alamat_orang_tua)

    pdf.cell(50, 10, 'Nomor Orang Tua', 0, 0)
    pdf.cell(0, 10, phone_orang_tua, ln=True)

    # Garis Pembatas
    pdf.ln(5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)

    # Pesan
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Pesan', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, pesan)
    
    # Hasil Deteksi
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Hasil Deteksi Kedipan', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, hasil_deteksi, ln=True)

    # Spasi sebelum kolom tanda tangan
    pdf.ln(3)


    # Simpan PDF dalam memori sebagai byte
    pdf_output = pdf.output(dest='S').encode('latin1')

    # Buat nama file PDF
    pdf_name = f"hasil_deteksi_{nama.replace(' ', '_')}.pdf"

    # Kirim file PDF untuk diunduh
    return send_file(io.BytesIO(pdf_output), as_attachment=True, download_name=pdf_name, mimetype='application/pdf')


# Fungsi untuk menangkap dan menampilkan video dari kamera
# def gen():
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()  # Baca frame dari kamera
#         if not ret or not camera_active:
#             break
#         ret, buffer = cv2.imencode('.jpg', frame)  # Encode frame sebagai gambar JPG
#         frame = buffer.tobytes()  # Ubah menjadi byte stream
#         yield (b'--frame\r\n'  # Kirim gambar dalam format streaming multipart
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#     cap.release()  # Tutup kamera

# # Route untuk video streaming
# @app.route('/video_feed')
# def video_feed():
#     return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Fungsi untuk menangkap dan menampilkan video dari kamera
def gen():
    cap = cv2.VideoCapture(0)  # Membuka kamera
    while True:
        ret, frame = cap.read()  # Baca frame dari kamera
        if not ret or not camera_active:
            break

        # Deteksi wajah pada frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Ubah ke grayscale
        faces = detector(gray)  # Deteksi wajah

        for face in faces:
            landmarks = predictor(gray, face)  # Prediksi landmark pada wajah
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Gambar titik landmark

        ret, buffer = cv2.imencode('.jpg', frame)  # Encode frame sebagai gambar JPG
        frame = buffer.tobytes()  # Ubah menjadi byte stream
        yield (b'--frame\r\n'  # Kirim gambar dalam format streaming multipart
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  # Tutup kamera

# Route untuk video streaming
@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Menjalankan aplikasi Flask
if __name__ == "__main__":
    app.run(debug=True)
