# Feature Extractors and Matchers

Визуальные SLAM/VO системы используют комбинацию **детектора**, **дескриптора** и **матчера**. Ниже приведён обзор основных алгоритмов, которые можно реализовать с помощью OpenCV (без внешних библиотек).

## 1. Детекторы

**Задача**: найти информативные точки (keypoints) на изображении.
**Выход**: координаты и характеристики ключевых точек.

* **Harris Corner** — базовый метод обнаружения углов.
* **Shi-Tomasi / GFTT (Good Features to Track)** — улучшение Harris, часто используется в KLT.
* **FAST (Features from Accelerated Segment Test)** — быстрый угловой детектор.
* **AGAST** — модификация FAST, более устойчивая.
* **SIFT (Scale-Invariant Feature Transform)** — устойчив к масштабу и повороту.
* **SURF (Speeded Up Robust Features)** — ускоренный вариант SIFT.
* **ORB (Oriented FAST and Rotated BRIEF)** — комбинация FAST и BRIEF, бесплатная альтернатива SIFT/SURF.
* **BRISK (Binary Robust Invariant Scalable Keypoints)** — быстрый бинарный детектор.
* **KAZE / AKAZE** — масштабно-инвариантные методы, работают в нелинейных масштабных пространствах.
* **MSER (Maximally Stable Extremal Regions)** — хорошо работает на текстовых и ярко выраженных объектах.
* **STAR Detector** — на основе центроида.

## 2. Дескрипторы

**Задача**: описать окрестность каждой ключевой точки в виде вектора.
**Выход**: набор векторов (обычно бинарных или вещественных).

* **SIFT** — 128-мерный float-дескриптор.
* **SURF** — 64-мерный float-дескриптор.
* **ORB** — бинарный дескриптор (256 бит).
* **BRIEF** — простой бинарный дескриптор (не инвариантен к повороту).
* **BRISK** — бинарный дескриптор.
* **FREAK (Fast Retina Keypoint)** — бинарный, имитирует структуру сетчатки.
* **KAZE / AKAZE** — поддерживают собственные дескрипторы.
* **DAISY** — плотный дескриптор для изображений.
* **LATCH** — бинарный дескриптор на основе трёх патчей.

## 3. Матчеры

**Задача**: сопоставить дескрипторы между кадрами.
**Выход**: пары совпадающих точек.

* **Brute-Force Matcher**

  * `cv2.BFMatcher(cv2.NORM_HAMMING)` — для бинарных дескрипторов (ORB, BRIEF, BRISK).
  * `cv2.BFMatcher(cv2.NORM_L2)` — для float-дескрипторов (SIFT, SURF).
* **FLANN (Fast Library for Approximate Nearest Neighbors)**

  * Используется для float-дескрипторов (SIFT, SURF).
  * Быстрее, чем BF, на больших выборках.
* **KLT (Lucas-Kanade Tracker)**

  * Использует оптический поток (`cv2.calcOpticalFlowPyrLK`) для трекинга ключевых точек между кадрами, без явных дескрипторов.


## 4. Комбинации

В SLAM/VO чаще всего используются следующие связки:

* **ORB detector + ORB descriptor + BFMatcher(Hamming)**
  Универсальный вариант, используется в ORB-SLAM.

* **SIFT detector + SIFT descriptor + FLANN(L2)**
  Более устойчивый, но медленный.

* **FAST detector + BRIEF descriptor + BFMatcher(Hamming)**
  Очень быстрый, подходит для real-time на слабых устройствах.

* **KAZE detector + KAZE descriptor + BFMatcher(L2)**
  Более качественный, но медленный.

* **Shi-Tomasi + KLT (Lucas-Kanade optical flow)**
  Классический трекер без дескрипторов, быстрый, но менее устойчивый к сильным изменениям сцены.
