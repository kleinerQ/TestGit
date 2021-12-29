from wondercise_data_core.file_io.record_io import Record
from bokeh.plotting import figure, output_file, show
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, lfilter

class WonderCoreCounter:
    def __init__(self, wcr_data_path):
        self.wcr_data = Record.load(pathlib.Path(wcr_data_path))

        device_sampling_rate = 100

        sampling_sec = 0.5

        axial_data = get_data(self.wcr_data, 'A')

        sampling_rate = int(device_sampling_rate * sampling_sec)

        reference_Ax = None
        reference_Ay = None
        reference_Az = None

        reference_diff_list = []

        count = 0

        state = 'still'

        max_direction = None

        min_direction = None

        for data in axial_data['left_wrist_band']:

            Ax = data[1]
            Ay = data[2]
            Az = data[3]

            """=== main algorithm ==="""

            """set 3 axial reference point"""
            if reference_Ax is None and reference_Ay is None and reference_Az is None:
                reference_Ax = Ax
                reference_Ay = Ay
                reference_Az = Az

            """accumulate enough data to detector"""
            if len(reference_diff_list) < sampling_rate:
                reference_diff_list.append([Ax - reference_Ax, Ay - reference_Ay, Az - reference_Az])

            else:
                reference_diff_list_T = np.array(reference_diff_list).T

                """calculate detection value from 3 axial, take the absolute value of each average and get sum"""
                detection_value = abs(np.mean(reference_diff_list_T[0])) + abs(np.mean(reference_diff_list_T[1])) + abs(
                    np.mean(reference_diff_list_T[2]))

                """search the max and min detection value"""
                if max_direction is None and min_direction is None:
                    max_direction = detection_value
                    min_direction = detection_value

                elif detection_value > max_direction:
                    max_direction = detection_value

                elif detection_value < min_direction:
                    min_direction = detection_value

                """state machine and count"""
                if state == 'still':
                    if detection_value > max_direction * 0.75:
                        state = 'action'

                if state == 'action':
                    if detection_value < (max_direction + min_direction) * 0.25:
                        state = 'still'
                        count += 1

                # print(data[0], detection_value, max_direction, min_direction)
                """result"""
                print(data[0], count)

                """renew regerence list"""
                reference_diff_list.pop(0)
                reference_diff_list.append([Ax - reference_Ax, Ay - reference_Ay, Az - reference_Az])

            """=== end ==="""


class SwayNFitCounter:
    def __init__(self, wcr_data_path):
        self.wcr_data = Record.load(pathlib.Path(wcr_data_path))

        axial_data = get_data(self.wcr_data, 'A')

        sampling_rate = 30

        delete_data = 15

        minimun_start_value = 0.02

        maximun_detection_value = 0.3

        detector_range = 2

        last_Ax = None
        last_Ay = None
        last_Az = None

        reference_diff_list = []

        max_direction = None
        min_direction = None

        state = 'still'

        right_detector = False

        left_detector = False

        count = 0

        for data in axial_data['left_wrist_band']:

            Ax = data[1]
            Ay = data[2]
            Az = data[3]

            """=== main algorithm ==="""

            if last_Ax is None and last_Ay is None and last_Az is None:
                last_Ax = Ax
                last_Ay = Ay
                last_Az = Az

            """diff from previous value and save in list"""
            if len(reference_diff_list) < sampling_rate:
                reference_diff_list.append([data[0], Ax - last_Ax, Ay - last_Ay, Az - last_Az])

            else:
                reference_diff_list_T = np.array(reference_diff_list).T

                """calculate each average and get sum"""
                detection_value = np.mean(reference_diff_list_T[1]) + np.mean(reference_diff_list_T[2]) + np.mean(
                    reference_diff_list_T[3])

                if max_direction is None and min_direction is None:
                    max_direction = detection_value
                    min_direction = detection_value

                else:
                    if max_direction < detection_value and max_direction < maximun_detection_value:
                        max_direction = detection_value
                    if min_direction > detection_value and min_direction > -maximun_detection_value:
                        min_direction = detection_value

                """state machine"""
                if state == 'still':
                    if detection_value < min_direction / detector_range and min_direction < -minimun_start_value:
                        state = 'bottom'
                        left_detector = True
                    elif detection_value > max_direction / detector_range and max_direction > minimun_start_value:
                        state = 'vertex'
                        right_detector = True
                elif state == 'vertex':
                    if detection_value < min_direction / detector_range and min_direction < -minimun_start_value:
                        state = 'bottom'
                        left_detector = True
                elif state == 'bottom':
                    if detection_value > max_direction / detector_range and max_direction > minimun_start_value:
                        state = 'vertex'
                        right_detector = True

                """counter"""
                if right_detector and left_detector:
                    count += 1
                    right_detector = False
                    left_detector = False

                    del reference_diff_list[0:delete_data]

                print(data[0], detection_value, state, count)

                reference_diff_list.pop(0)
                reference_diff_list.append([data[0], Ax - last_Ax, Ay - last_Ay, Az - last_Az])

            last_Ax = Ax
            last_Ay = Ay
            last_Az = Az

            """=== end ==="""

class Genius:
    def __init__(self, wcr_data_path):
        self.wcr_data = Record.load(pathlib.Path(wcr_data_path))

        axial_data = get_data(self.wcr_data, 'G')

        detector_rate = 30

        detector_list = []

        max_detector = None
        min_detector = None

        count = 0

        top_tf = None
        down_tf = None

        state = 'still'

        for data in axial_data['right_wrist_band']:
            Gx = data[1]
            Gy = data[2]
            Gz = data[3]

            """=== main algorithm ==="""

            if len(detector_list) < detector_rate:
                detector_list.append([data[0], Gx, Gy, Gz])

            else:
                detector_list_T = np.array(detector_list).T

                detector = np.mean(detector_list_T[1]) + np.mean(detector_list_T[2]) + np.mean(detector_list_T[3])

                if max_detector is None and min_detector is None:
                    max_detector = detector
                    min_detector = detector

                """found max and min detector and restrict it"""
                if detector > max_detector and 50 > detector > 20:
                    max_detector = detector

                if detector < min_detector and -50 < detector < -20:
                    min_detector = detector

                """state machine"""
                if state == 'still' or state == 'down':
                    if detector > 10 and detector > max_detector * 0.8:
                        state = 'top'
                        top_tf = True

                elif state == 'top':
                    if detector < -10 and detector < min_detector * 0.8:
                        state = 'down'
                        down_tf = True

                """counter and reset"""
                if top_tf and down_tf:
                    count += 1
                    top_tf = False
                    down_tf = False

                print(data[0], detector, max_detector, min_detector, state)

                if len(detector_list) == detector_rate:
                    detector_list.pop(0)
                    detector_list.append([data[0], Gx, Gy, Gz])

                """===end==="""

        print(count)


class SlideFit:
    def __init__(self, wcr_data_path):
        self.wcr_data = Record.load(pathlib.Path(wcr_data_path))

        axial_data = get_data(self.wcr_data, 'A')

        detector_rate = 50

        detector_list = []

        max = None
        min = None

        reference_Ax = None
        reference_Ay = None
        reference_Az = None

        count = 0

        state = 'still'

        for data in axial_data['right_wrist_band']:
            Ax = data[1]
            Ay = data[2]
            Az = data[3]

            """=== main algorithm ==="""

            """set 3 axial reference point"""

            if reference_Ax is None and reference_Ay is None and reference_Az is None:
                reference_Ax = Ax
                reference_Ay = Ay
                reference_Az = Az

            """save diff from point to reference"""
            if len(detector_list) < detector_rate:
                detector_list.append([data[0], (Ax - reference_Ax), (Ay - reference_Ay), (Az - reference_Az)])

            else:
                detector_list_T = np.array(detector_list).T

                detector = np.mean(detector_list_T[1]) + np.mean(detector_list_T[2]) + np.mean(detector_list_T[3])

                """set max and min detector and restrict it"""
                if 0.2 > detector > 0.1 or -0.2 < detector < -0.1:

                    if max is None and min is None:
                        max = detector
                        min = detector

                    elif detector > max:
                        max = detector

                    elif detector < min:
                        min = detector

                """determine the state, state machine"""
                if max is not None and min is not None:
                    if state == 'still':
                        if detector > max / 2:
                            state = 'top'

                    elif state == 'top':
                        if detector < min / 2:
                            state = 'down'
                            count += 1
                            detector_list.clear()

                    elif state == 'down':
                        if detector > max / 2:
                            state = 'top'

                if len(detector_list) == detector_rate:
                    detector_list.pop(0)
                    detector_list.append([data[0], (Ax - reference_Ax), (Ay - reference_Ay), (Az - reference_Az)])

        print(count)

        """===end==="""

class GeniusPushUp:
    def __init__(self, wcr_data_path):
        self.wcr_data = Record.load(pathlib.Path(wcr_data_path))

        axial_data = get_data(self.wcr_data, 'G')

        detector_rate = 30

        detector_list = []

        max = None
        min = None

        reference_Gx = None
        reference_Gy = None
        reference_Gz = None

        top_tf = False
        down_tf = False

        count = 0

        state = 'still'

        for data in axial_data['right_wrist_band']:
            Gx = data[1]
            Gy = data[2]
            Gz = data[3]

            """=== main algorithm ==="""

            """set 3 axial reference point"""

            if reference_Gx is None and reference_Gy is None and reference_Gz is None:
                reference_Gx = Gx
                reference_Gy = Gy
                reference_Gz = Gz

            if len(detector_list) < detector_rate:
                detector_list.append([data[0], (Gx - reference_Gx), (Gy - reference_Gy), (Gz - reference_Gz)])

            else:
                detector_list_T = np.array(detector_list).T

                detector = np.mean(detector_list_T[1]) + np.mean(detector_list_T[2]) + np.mean(detector_list_T[3])

                """set max and min detector and restrict it"""

                if max is None and min is None:
                    max = detector
                    min = detector

                elif detector > max and 100 > detector > 40:
                    max = detector

                elif detector < min and -100 < detector < -40:
                    min = detector

                """determine the state, state machine"""

                if abs(detector) > 10:
                    if state == 'still':
                        if detector > max / 2:
                            state = 'top'
                            top_tf = True
                        elif detector < min / 2:
                            state = 'down'
                            down_tf = True

                    elif state == 'top':
                        if detector < min / 2:
                            state = 'down'
                            down_tf = True

                    elif state == 'down':
                        if detector > max / 2:
                            state = 'top'
                            top_tf = True

                if top_tf and down_tf:
                    count += 1
                    top_tf = False
                    down_tf = False

                print(data[0], max, min, detector, state, count)

                detector_list.pop(0)
                detector_list.append([data[0], (Gx - reference_Gx), (Gy - reference_Gy), (Gz - reference_Gz)])

        print(count)

        """end"""

class PushUp:
    def __init__(self, test_wcr_path):
        self.wcr_path = pathlib.Path(test_wcr_path)
        self.wcr_data = Record.load(self.wcr_path)

        A_data = np.array(self.wcr_data.df['unoriented_chest_belt'].iloc[:, :3])
        # G_data = np.array(self.wcr_data.df['unoriented_chest_belt'].iloc[:, 3:6])

        filtered_data_list = Butter_LosspassFilter(A_data)
        # G_filtered_data_list = Butter_LosspassFilter(G_data)

        # Ax = filtered_data_list[:, 0]
        # Ay = filtered_data_list[:, 1]
        Az = filtered_data_list[:, 2]

        # A = Az

        tmp = []

        tmp_limit = 20

        A_x = [i for i in range(len(Az))]

        # for i in range(0, len(Az)):
        #     if i == 0:
        #         tmp.append(Az[i] - A[i])
        #     else:
        #         tmp.append(Az[i] - A[i-1])

        index = 0

        ready_threhold = [0.8, 1.2]

        status = 'unready'

        top_check = False
        down_check = False

        count = 0

        for data in Az:
            if len(tmp) < tmp_limit:
                tmp.append(data)

            else:
                tmp.pop(0)
                tmp.append(data)

            judgment_value = np.array(tmp).mean()

            if ready_threhold[0] < judgment_value < ready_threhold[1] and status == 'unready':
                status = 'ready'
            elif status == 'ready':
                if judgment_value > ready_threhold[1]:
                    status = 'top'
                elif judgment_value < ready_threhold[0]:
                    status = 'down'
            elif status == 'top' and judgment_value < ready_threhold[0]:
                status = 'down'
                down_check = True

            elif status == 'down' and judgment_value > ready_threhold[1]:
                status = 'top'
                top_check = True

            if down_check and top_check:
                count += 1
                top_check = False
                down_check = False

            print(index, judgment_value, count)
            index += 1

        # tmp = tmp
        #
        # output_file('pushup.html')
        #
        # p = figure(plot_width=1280, plot_height=700)
        #
        # p.line([i for i in range(len(tmp))], tmp, line_width=2)
        #
        # p.line(A_x, Az, line_width=2, color='red')
        #
        # show(p)

def Butter_LosspassFilter(data):
    # plot_FFT_frequence(data[:, 2], 'Ay')
    global a, b
    m, n = data.shape[0], data.shape[1]
    V = np.full((m, n), 0, dtype=np.float64)
    for i in range(n):
        cutoff = 2
        order = 3
        nyq = 0.5 * 100
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, data[:, i])
        V[:, i] = y

    # print(a, b)

    return V

def plot_FFT_frequence(data, axis):
    sp = np.fft.fft(data)
    ampSP = np.abs(sp)
    time_step = 1. / 100
    freqs = np.fft.fftfreq(sp.size, time_step)
    idx = np.argsort(freqs)
    color = ['red', 'navy', 'green', 'fuchsia', 'black', 'lime', 'brown', 'royalblue', 'yellow', 'cyan']
    p = figure(tools="pan, wheel_zoom, box_select, reset, hover, save", plot_width=1600, plot_height=900,
               active_scroll='wheel_zoom')
    p.line(freqs[idx[len(idx) // 2:]], ampSP[idx[len(idx) // 2:]], line_width=2, color=color[0],
           legend_label=f'FFT_frequence_{axis}')
    p.legend.click_policy = "hide"
    show(p)

def get_data(wcr_data, axial):
    data = {}

    for type in wcr_data.device_types:
        device_data = []

        if axial == 'A':
            for index in wcr_data.df[type].index:
                device_data.append([index, wcr_data.df[type]['Ax'][index], wcr_data.df[type]['Ay'][index],
                                    wcr_data.df[type]['Az'][index]])

        elif axial == 'G':
            for index in wcr_data.df[type].index:
                device_data.append([index, wcr_data.df[type]['Gx'][index], wcr_data.df[type]['Gy'][index],
                                    wcr_data.df[type]['Gz'][index]])

        data.update({type: device_data})

    return data


# WonderCoreCounter('/home/algorithm/下載/Smart/20210817_test data_Smart_Yadi_01_02.wcr')
# WonderCoreCounter('/home/algorithm/下載/WonderCore2/20210817_test data_WonderCore2_KC_01_02.wcr')
# SwayNFitCounter('/home/algorithm/下載/Sway n Fit/20210817_test data_Sway n Fit_KC_01_02.wcr')
# Genius('/home/algorithm/Denniel/Data/Record_Data/courses/test data/rerecord/Workout 07/20210823_test data_Genius_Tian_01_02.wcr')
# SlideFit('/home/algorithm/Denniel/Data/Record_Data/courses/test data/rerecord/Workout 08/20210817_test data_Slide fit_KH_01_02.wcr')
# GeniusPushUp('/home/algorithm/Denniel/Data/Record_Data/courses/test data/rerecord/Workout 11/20210826_test data_Genius push up_慶豐_02_02.wcr')
PushUp('/home/algorithm/Denniel/Data/PushUpData/20211027_test data_Push up_俞淵_02_02.wcr')