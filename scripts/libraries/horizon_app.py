"""
horizon_app.py - OpenMV H7 Plus horizon / nadir sensor, built into the firmware.
Started by boot.py. Frozen together with horizon_core.py and nadir_fit.py.

BEFORE FLIGHT you must set CAM_FX for the lens/sensor/resolution you really fly.
  fx_px = (WIDTH_PX / 2) / tan(HFOV / 2)        (HFOV measured, see notes)
Changing any setting here means rebuilding and reflashing the firmware.
Nothing here has been run on hardware yet - OpenMV-specific calls are marked  # HW
"""
import sensor, image, pyb, gc, time, os
from machine import UART
from horizon_core import estimate, cone_alpha, nad_line, horz_line, limb_curve

# ============================== CONFIG ======================================
CAM_FX     = 150.0        # px at 320x240. OpenMV ultra-wide 1.7mm + OV5640: HFOV 93.7 deg
                          # -> 160/tan(46.85deg) = 150. At 640 wide use 300. Refine by calibration.
CAM_K1     = 0.0          # radial distortion from calibrate_lens.py (scaled to this resolution
CAM_K2     = 0.0          # is NOT needed: k1,k2 act on normalised coords). 0 = pinhole, WRONG
                          # for this lens near the edges until calibrated.
FRAME      = sensor.QVGA  # 320x240. If you change this, change CAM_FX proportionally
ALT_KM     = 400.0
SPACE_THR  = 30           # gray level below which a pixel counts as space
CONFIRM    = 3            # bright run (px) needed after space; rejects stars
PERIOD_MS  = 3000

TEST_DIR   = None         # e.g. "/sdcard/test": process saved .pgm/.jpg files, then stop
RESULT_DIR = "/flash/results"  # test mode writes <name>_fit.jpg + results.txt here
SAVE_FITTED = False       # live mode: also save a fitted image every frame (flash wear!)
SAVE_IMAGES = False       # saving a JPEG every frame wears flash and costs time
SEND_IMAGE  = False       # raw image over UART: slow, leave off
# ============================================================================

LED_R, LED_G, LED_B = 1, 2, 3
ALPHA = cone_alpha(ALT_KM)

uart = UART(3, 115200)
uart.init(115200, bits=8, parity=None, stop=1)


def led(i, on):
    try:
        pyb.LED(i).on() if on else pyb.LED(i).off()
    except Exception:
        pass


def gray_buf(img):                                                     # HW
    try:
        return img.bytearray()          # reference to the frame buffer, no copy
    except Exception:
        return bytearray(img)


def process(img, counter, name=""):
    led(LED_G, True)
    t0 = time.ticks_ms()
    W, H = img.width(), img.height()
    buf = gray_buf(img)
    res = estimate(buf, W, H, CAM_FX, ALPHA, thr=SPACE_THR, confirm=CONFIRM,
                   k1=CAM_K1, k2=CAM_K2)
    dt = time.ticks_diff(time.ticks_ms(), t0)
    print("%s %s  W=%d fx=%.0f  %dms" % (name, nad_line(res), W, CAM_FX, dt))
    try:
        uart.write(horz_line(counter, res).encode())
    except Exception as e:
        print("uart:", e)
    led(LED_G, False)
    return res


def save_fit(img, res, name):
    """Draw the fit onto the image and save it: bright cross = kept point, dim = rejected,
    bright line = fitted limb. Same curve maths as the laptop overlay (limb_curve)."""
    try:
        pts, mask = res["pts"], res["mask"]
        for i, (x, y) in enumerate(pts):
            ok = mask is not None and i < len(mask) and mask[i]
            img.draw_cross(int(x), int(y), color=255 if ok else 90, size=3 if ok else 1)  # HW
        if res["n"] is not None:
            prev = None
            for p in limb_curve(res["n"], ALPHA, CAM_FX, img.width(), img.height(), 90):
                cur = None if p is None else (int(p[0]), int(p[1]))
                if prev and cur:
                    img.draw_line(prev[0], prev[1], cur[0], cur[1], color=255)      # HW
                prev = cur
        img.save(RESULT_DIR + "/" + name + "_fit.jpg")                              # HW
    except Exception as e:
        print("save_fit:", e)


# ------------------------------ test mode -----------------------------------
if TEST_DIR:
    try:
        os.mkdir(RESULT_DIR)
    except Exception:
        pass                                    # already exists
    files = sorted(f for f in os.listdir(TEST_DIR)
                   if f.lower().endswith((".pgm", ".jpg", ".jpeg", ".png", ".bmp")))
    print("TEST MODE:", len(files), "files in", TEST_DIR, "-> results in", RESULT_DIR)
    for i, fn in enumerate(files):
        gc.collect()
        img = image.Image(TEST_DIR + "/" + fn, copy_to_fb=True)        # HW
        img.to_grayscale()                                             # HW
        res = process(img, i + 1, fn[:24])
        try:
            with open(RESULT_DIR + "/results.txt", "a") as fh:
                fh.write(fn[:24] + " " + nad_line(res) + "\n")
        except Exception as e:
            print("results.txt:", e)
        save_fit(img, res, fn.rsplit(".", 1)[0][:24])
    print("TEST DONE - fitted images in", RESULT_DIR)
    raise SystemExit

# ------------------------------ live mode -----------------------------------
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(FRAME)
sensor.skip_frames(time=2000)
# For night work lock exposure/gain instead of auto (tune on the ground):
# sensor.set_auto_gain(False); sensor.set_auto_exposure(False, exposure_us=...)   # HW
gc.collect()

counter = 1
while True:
    pyb.delay(PERIOD_MS)
    led(LED_R, True)
    img = sensor.snapshot()
    led(LED_R, False)
    if SAVE_IMAGES:
        try:
            img.save("HORIZON_%d.jpg" % counter)
        except Exception as e:
            print("save:", e)
    res = process(img, counter)
    if SAVE_FITTED:
        try:
            os.mkdir(RESULT_DIR)
        except Exception:
            pass
        save_fit(img, res, "C%d" % counter)
    if SEND_IMAGE and res["status"] == "OK":
        try:
            img.save("send_tmp.jpg")
            with open("send_tmp.jpg", "rb") as f:
                uart.write(f.read())
        except Exception as e:
            print("send:", e)
    gc.collect()
    counter += 1
