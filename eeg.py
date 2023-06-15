import matplotlib
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
from scipy import signal

matplotlib.use('TkAgg')


# Вспомогательная функция - вычисление БПФ сигнала с частотой дискретизации fs
# и сглаживающим окном по желанию
def FFT(x, fs, window=None):
    w = np.array([1 for i in range(len(x))])
    if window is not None:
        w = signal.get_window(window, len(x))
    x -= np.mean(x)
    N = len(x)
    X = 2 / N * fft.fft(x * w, n=max(fs, N))
    N = max(fs, N)
    f = np.linspace(0, fs // 2, N // 2)
    return f, abs(X[0:N // 2]) ** 2


# Вспомогательная функция - вычисление спектрограммы сигнала с
# количеством отрезков segments, с наложением в половину длины отрезка,
# частоты вырезаются только из диапазона [min_hz, max_hz], сглаживающее окно - по желанию
def spectrogram(x, fs, segments=20, min_hz=0, max_hz=40, window=None):
    step = len(x) // segments
    step //= 2
    segments_with_overlap = segments * 2
    res = []
    f, _ = FFT(x[0:2*step], fs)
    mx = max(np.argwhere(f <= max_hz).flatten())
    mn = min(np.argwhere(f >= min_hz).flatten())
    f = f[mn:mx+1]
    for i in range(segments_with_overlap):
        tmp_x = x[i * step:(i + 2) * step]
        _, tmp_X = FFT(tmp_x, fs, window)
        res.append(tmp_X[mn:mx + 1])
    return f, np.array(res)


# Вспомогательная функция - вычисление энтропии сигнала
def entropy(signal):
    n = len(signal)
    ans = 0
    values, counts = np.unique(np.array(signal), return_counts=True)
    for i in range(len(values)):
        p = counts[i] / n
        ans -= p * np.log2(p)
    return ans


# Вспомогательная функция - нахождение максимальной амплитуды
def max_amplitude(spectrum):
    return max(spectrum)


# Вспомогательная функция - нахождение частоты,
# на которой достигается максимальная амплитуда
def argmax_amplitude(freq, spectrum):
    return freq[np.argmax(spectrum)]


# Вспомогательная функция - подсчет среднего значения в спектре
def avg_amplitude(spectrum):
    return np.average(spectrum)


# Вспомогательная функция - вычисление средневзвешенной частоты
def expected_freq(freq, spectrum):
    return np.sum(
        [spectrum[i] * freq[i] for i in range(len(freq))]) / np.sum(
        [spectrum[i] for i in range(len(freq))])


# Вспомогательная функция - подсчет интеграла методом прямоугольников
# на промежутке ends функции spectrum
def area(ends, spectrum, freq):
    a, b = ends
    mx = max(np.argwhere(freq <= b).flatten())
    mn = min(np.argwhere(freq >= a).flatten())
    res = 0
    step = freq[1] - freq[0]
    for x in range(mn, mx + 1):
        res += spectrum[x]
    return res * step


# Вспомогательная функция - подсчет мощностей ритмов
def power_rhythms(freq, spectrum):
    gamma = (32, 100)
    gamma_area = area(gamma, spectrum, freq)
    beta = (13, 32)
    beta_area = area(beta, spectrum, freq)
    alpha = (8, 13)
    alpha_area = area(alpha, spectrum, freq)
    theta = (4, 8)
    theta_area = area(theta, spectrum, freq)
    delta = (1, 4)
    delta_area = area(delta, spectrum, freq)
    return [delta_area, theta_area, alpha_area, beta_area, gamma_area]


# Вычисление наиболее выраженного ритма
def powerful_rhythm(freq, spectrum):
    rhythms = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    return rhythms[np.argmax(power_rhythms(freq, spectrum))]


# Вычисление энтропии сигнала во времени
# (вычисляется для сигнала целиком, а не кусочка по частотам)
def calc_total_entropy(x, fs, segments):
    step = len(x) // segments
    step //= 2
    segments_with_overlap = segments * 2
    res = []
    for i in range(segments_with_overlap):
        tmp_x = x[i * step:(i + 2) * step]
        res.append(entropy(tmp_x))
    return res


# Вычисление мощности частот во времени
# (т.е. интеграл спектра на отрезке [min_hz, max_hz])
def calc_area_of_rhythm(x, fs, segments, min_hz, max_hz):
    step = len(x) // segments
    step //= 2
    segments_with_overlap = segments * 2
    res = []
    for i in range(segments_with_overlap):
        tmp_x = x[i*step:(i+2)*step]
        freq, X = FFT(tmp_x, fs)
        res.append(area((min_hz, max_hz), X, freq))
    return res


# Вспомогательная функция - поиск вспышек на частотах
# [min_hz, max_hz] для одного электрода
def find_burst_of_rhythm(x, fs, segments, min_hz, max_hz, alpha):
    areas = calc_area_of_rhythm(x, fs, segments, min_hz, max_hz)
    avr = np.average(areas)
    return np.array([i for i in range(len(areas)) if areas[i] >= avr * alpha])


# Вспомогательная функция - нахождение разности между первым и вторым списком
def find_diff(res_x, res_y):
    set_x = set(res_x)
    set_y = set(res_y)
    return np.array(sorted(list(set_x.difference(set_y))))


# Вспомогательная функция - перевод индексов в массиве времени в секунды
def index_to_seconds(index, n, segments, fs):
    step = n // segments
    step //= 2
    return step * index / fs


# Изучение тех эпох, на которых мощность на диапазоне частот
# [min_hz, max_hz] была больше чем alpha * среднюю мощность (назовем это вспышкой),
# затем отбор только тех "вспышек", которые присутствуют только в первом сигнале
def find_relative_burst(x, y, fs, segments, min_hz, max_hz, alpha=2.0):
    res_x = find_burst_of_rhythm(x, fs, segments, min_hz, max_hz, alpha)
    res_y = find_burst_of_rhythm(y, fs, segments, min_hz, max_hz, alpha)
    return np.array(list(map(lambda el: index_to_seconds(el, len(x), segments, fs),
             find_diff(res_x, res_y))))


# Получение всех спектральных характеристик во времени из спектрограммы
def all_info(x, fs, segments=20, min_hz=0, max_hz=40):
    f, res = spectrogram(x, fs, segments=segments, min_hz=min_hz, max_hz=max_hz)
    total_entropy = calc_total_entropy(x, fs, segments)
    total_max_amplitude = list(map(max_amplitude, res))
    total_argmax_amplitude = list(map(lambda el: argmax_amplitude(f, el), res))
    total_avg_amplitude = list(map(avg_amplitude, res))
    total_expected_freq = list(map(lambda el: expected_freq(f, el), res))

    freq, X = spectrogram(x, fs, segments=segments, min_hz=0, max_hz=100)
    total_powerful_rhythm = list(map(lambda el: powerful_rhythm(freq, el), X))

    epochs = np.linspace(0, len(x)/fs, segments*2)

    return epochs, f, total_entropy, total_powerful_rhythm, total_max_amplitude, total_argmax_amplitude, \
           total_argmax_amplitude, total_avg_amplitude, total_expected_freq


# Функция для отрисовки спектрограммы в окне по частотам [mih_hz, max_hz]
def draw_spectrogram(x, fs, segments=20, min_hz=0, max_hz=30, axis=plt, window='blackman'):
    _, res = spectrogram(x, fs, segments=segments, min_hz=min_hz, max_hz=max_hz, window=window)
    res = res.T
    vmax = max(res.flatten())
    im = axis.imshow(res, origin='lower', extent=(0, len(x)//fs, min_hz, max_hz),
                    cmap=plt.colormaps['turbo'],
                    aspect=7, vmin=0, vmax=vmax)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    if axis == plt:
        plt.xlabel('Time')
        plt.ylabel('Hz')
    else:
        axis.set_xlabel('Time')
        axis.set_ylabel('Hz')


# Вспомогательная функция - нарисовать две спектрограммы рядом
def compare_two_spectorgram(x, y, fs, segments=20, min_hz=0, max_hz=30):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.suptitle('Сравнение двух спектрограмм')
    draw_spectrogram(x, fs, segments, min_hz, max_hz, ax1)
    draw_spectrogram(y, fs, segments, min_hz, max_hz, ax2)
    plt.show()


# Нахождение разницы между двумя спектрограммами
def diff_two_spectrogram(x, y, fs, segments=20, min_hz=0, max_hz=30):
    f, X = spectrogram(x, fs, segments, min_hz, max_hz)
    _, Y = spectrogram(y, fs, segments, min_hz, max_hz)
    diff = (np.array(X) - np.array(Y)).T
    vmax = max(diff.flatten())
    vmin = min(diff.flatten())
    plt.title('Разность двух спектрограмм')
    im = plt.imshow(diff, origin='lower', extent=(0, len(x)//fs, min_hz, max_hz),
                    cmap=plt.colormaps['plasma'],
                    aspect=10, vmin=vmin, vmax=vmax)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.xlabel('Time')
    plt.ylabel('Hz')
    plt.show()


# Вспомогательная функция - отрисовка двух графиков рядом
def draw_two_graphs(time, data_x, data_y, title, xlabel, ylabel):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(title)
    ax1.plot(time, data_x, color='#a44703')
    ax2.plot(time, data_y, color='#0647f4')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()


# Отрисовка сигнала начиная с секунды start, продолжительностью duration
def draw_signal(signal, fs, start, duration):
    time = np.linspace(0, len(signal)/fs, len(signal))
    a = int(start * fs)
    b = a + int(duration * fs)
    plt.title(f'Отрисовка сигнала длины {duration} сек. начиная с {start} сек.')
    plt.plot(time[a:b+1], signal[a:b+1], color='#0647f4')
    plt.xlabel('Time')
    plt.ylabel('Amplitude (dB)')
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()


# Отрисовка двух сигналов на одном графике
def draw_two_signals(signal_x, signal_y, fs, start, duration):
    time = np.linspace(0, len(signal_x)/fs, len(signal_x))
    a = int(start * fs)
    b = a + int(duration * fs)
    plt.title(f'Отрисовка сигнала длины {duration} сек. начиная с {start} сек.')
    plt.plot(time[a:b+1], signal_x[a:b+1], color='#0647f4')
    plt.plot(time[a:b+1], signal_y[a:b+1], color='#af0604')
    plt.xlabel('Time')
    plt.ylabel('Amplitude (dB)')
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()


# Обработка данных спектрограммы для двух электродов,
# вычисление спектральных характеристик во времени и их сравнение
def compare_two_all_info(x, y, fs, segments=20, min_hz=0, max_hz=30):
    epochs, f, total_entropy_x, total_powerful_rhythm_x, \
    total_max_amplitude_x, total_argmax_amplitude_x, \
    total_argmax_amplitude_x, total_avg_amplitude_x, total_expected_freq_x = all_info(x, fs, segments, min_hz, max_hz)
    _, _, total_entropy_y, total_powerful_rhythm_y, \
    total_max_amplitude_y, total_argmax_amplitude_y, \
    total_argmax_amplitude_y, total_avg_amplitude_y, total_expected_freq_y = all_info(y, fs, segments, min_hz, max_hz)

    draw_two_graphs(epochs, total_powerful_rhythm_x, total_powerful_rhythm_y, 'Наиболее выраженные ритмы', 'Time', 'Rhythms')
    draw_two_graphs(epochs, total_expected_freq_x, total_expected_freq_y, 'Мат. ожидание частоты', 'Time', 'Frequency (Hz)')
    draw_two_graphs(epochs, total_max_amplitude_x, total_max_amplitude_y, 'Наибольшая амплитуда', 'Time', 'Amplitude (dB)')
    draw_two_graphs(epochs, total_avg_amplitude_x, total_avg_amplitude_y, 'Средняя амплитуда', 'Time', 'Amplitude (dB)')
    draw_two_graphs(epochs, total_entropy_x, total_entropy_y, 'Энтропия сигнала (целиком)', 'Time', 'Entropy (bits)')
    draw_two_graphs(epochs, total_argmax_amplitude_x, total_argmax_amplitude_y, 'Частота, на которой достигается максимальная амплитуда', 'Time', 'Frequency (Hz)')


# Считывание файла в формате прямоугольной матрицы,
# где один столбец соответствует одному электроду
def read_file(filename: str):
    file = open(filename, mode='r', encoding='UTF-8')
    res = []
    for line in file:
        res.append(list(map(float, line.split())))
    file.close()
    return np.array(res).T


# Пример работы с библиотекой MNE
def mne_analysis():
    sample_data_folder = mne.datasets.sample.data_path()
    print(sample_data_folder)
    sample_data_raw_file = (
            sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
    )
    raw = mne.io.read_raw_fif(sample_data_raw_file)
    print(raw.info)

    X = raw.compute_psd(tmin=10, tmax=20, fmin=1, fmax=45, picks="eeg")
    X.plot(picks='data', exclude='bads')
    raw.plot(block=True)
    X.plot_topomap(ch_type="eeg", agg_fun=np.median)

    # set up and fit the ICA
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw)
    ica.exclude = [1, 2]  # details on how we picked these are omitted here
    ica.plot_properties(raw, picks=ica.exclude)

    orig_raw = raw.copy()
    raw.load_data()
    ica.apply(raw)

    # show some frontal channels to clearly illustrate the artifact removal
    chs = [
        "MEG 0111",
        "MEG 0121",
        "MEG 0131",
        "MEG 0211",
        "MEG 0221",
        "MEG 0231",
        "MEG 0311",
        "MEG 0321",
        "MEG 0331",
        "MEG 1511",
        "MEG 1521",
        "MEG 1531",
        "EEG 001",
        "EEG 002",
        "EEG 003",
        "EEG 004",
        "EEG 005",
        "EEG 006",
        "EEG 007",
        "EEG 008",
    ]
    chan_idxs = [raw.ch_names.index(ch) for ch in chs]
    orig_raw.plot(order=chan_idxs, start=12, duration=4)
    raw.plot(order=chan_idxs, start=12, duration=4, block=True)

    events = mne.find_events(raw, stim_channel="STI 014")
    print(events[:5])  # show the first 5
    event_dict = {
        "auditory/left": 1,
        "auditory/right": 2,
        "visual/left": 3,
        "visual/right": 4,
        "smiley": 5,
        "buttonpress": 32,
    }
    fig = mne.viz.plot_events(
        events, event_id=event_dict, sfreq=raw.info["sfreq"], first_samp=raw.first_samp
    )

    reject_criteria = dict(
        mag=4000e-15,  # 4000 fT
        grad=4000e-13,  # 4000 fT/cm
        eeg=150e-6,  # 150 µV
        eog=250e-6,
    )  # 250 µV
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=-0.2,
        tmax=0.5,
        reject=reject_criteria,
        preload=True,
    )
    conds_we_care_about = ["auditory/left", "auditory/right", "visual/left",
                           "visual/right", "buttonpress"]
    epochs.equalize_event_counts(conds_we_care_about)  # this operates in-place
    aud_epochs = epochs["auditory"]
    vis_epochs = epochs["visual"]
    button_epochs = epochs["buttonpress"]

    aud_epochs.plot_image(picks=['EEG 007', 'EEG 015', 'EEG 023'])

    epochs_spectrum = aud_epochs[0].compute_psd(picks='eeg')
    epochs_spectrum.plot(picks='data', exclude='bads')

    raw.plot(block=True)


# Точка старта программы
def main():
    fs = 500
    x = read_file('examples/raw1.txt')[11]
    y = read_file('examples/raw1.txt')[4]
    t = np.linspace(0, len(x)/fs, len(x))

    segments = 80
    min_hz = 8
    max_hz = 12

    draw_two_signals(x, y, fs, 130, 20)

    diff_two_spectrogram(x, y, fs, segments, min_hz, max_hz)
    compare_two_spectorgram(x, y, fs, segments, min_hz, max_hz)
    compare_two_all_info(x, y, fs, segments, min_hz, max_hz)
    print(find_relative_burst(x, y, fs, segments, min_hz, max_hz, alpha=1.5))
    print(find_relative_burst(y, x, fs, segments, min_hz, max_hz, alpha=1.5))


if __name__ == '__main__':
    # mne_analysis()
    main()
