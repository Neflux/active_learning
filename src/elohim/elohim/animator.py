from functools import partial
from functools import partial
from math import sin, cos, sqrt
from pathlib import Path

import matplotlib
import matplotlib.colors as clr
import matplotlib.colors as mcolors
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from dataset_ml import transform_function
from models import initialize_model
from utils import cv_from_binary, plot_transform, scaled_full_robot_geometry, mktransf, bytes_from_str, \
    predictions_from_dataframe

matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib import axes, figure, transforms, animation, gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance
from tqdm import tqdm

from typing import List

tqdm.pandas()

import config

d45 = np.pi / 4


class Animator:
    def __init__(self, df, static_targets, rate=30, save_path=None, s=None, extra_info=None):

        """Given a dataframe and a list of coordinates of static targets, builds a FuncAnimation.
            The animations shows the top down view, and the live camera feed (ax1 and ax2).
            TODO: occupancy map ax3

            The top down view also features the unprecise odometry, but displays it only when the difference from
                ground truth is large enough (blue: ground truth, orange: displaced odometry)

            Args:
                    @param df:  the merged dataframe that contains all the necessary data (sensors and odometry)
                    @param static_targets:  a list of sensors' readings.
                    @param rate: rate of the animation
                    @param save_path: if specified, the animation will be saved at this path instead of being shown
            """
        # rc('axes', unicode_minus=False)
        # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        plt.close()

        self.rate = rate

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=rate, metadata=dict(artist='Me'), bitrate=1800)

        if s is not None:
            df = df.iloc[s]

        frames = len(df)

        # Cache

        # self.sensor_readings = df['sensor'].to_numpy()
        self.odom_pos_data = df[['x', 'y', 'theta']].to_numpy()
        self.gt_pos_data = df[['ground_truth_odom_x', 'ground_truth_odom_y', 'ground_truth_odom_theta']].to_numpy()
        self.images = df['image'].map(cv_from_binary).map(partial(Image.fromarray, mode="RGB")).to_numpy()
        # self.sensor_data = df['sensor'].round(3).astype(str).values

        self.run_counter = df['run'].to_numpy()
        rd = df.groupby('run')['x'].count().astype(int)
        rd = pd.DataFrame({'len': rd, 'end': rd.cumsum().astype(int)})
        self.run_dict = rd.to_dict()

        self.omaps = np.array([np.array(x).reshape(20, 20) for x in df['target_map'].values])
        self.colored_omaps = []
        for x in self.omaps:
            res = np.empty((20, 20, 3), dtype=np.uint8)
            res[x == -1] = (190, 190, 190)
            res[x == 0] = (0, 255, 0)
            res[x == 1] = (255, 0, 0)
            self.colored_omaps.append(res)

        # self.colored_omaps = self.omaps
        self.fov_segments = 40

        def gradient_line(cmap):
            """ Returns a LineCollection composition that results in a single line colored with a gradient """
            gradient = np.linspace(0.0, 1.0, self.fov_segments)
            my_cmap = cmap(np.arange(cmap.N))
            my_cmap[:, -1] = np.linspace(1, 0, cmap.N)
            my_cmap = ListedColormap(my_cmap)
            return LineCollection([], array=gradient, cmap=my_cmap, linewidth=1.2)

        self.length = 10

        def gradient_area(color):
            """ Returns an AxesImage and its Polygon perimeter for the Thymio FOV """
            gradient = np.empty((100, 1, 4), dtype=float)
            rgb = mcolors.colorConverter.to_rgb(color)
            gradient[:, :, :3] = rgb
            gradient[:, :, -1] = np.linspace(0., .1, 100)[:, None]

            ext = [-self.length / sqrt(2), self.length / sqrt(2), -self.length / sqrt(2), 0]
            fov_area = ax1.imshow(gradient, aspect='auto', extent=ext, origin='lower')

            clip_path = Polygon(np.empty((3, 2)), fill=False, alpha=0)
            return fov_area, clip_path

        # rc('font', **{'family': 'serif', 'serif': ['Sathu']})  # Hiragino Maru Gothic Pro

        if 'predicted_map' in df.columns:
            #  # type:figure.Figure, List[axes.Axes]
            # import matplotlib.gridspec as gridspec
            # fig = plt.figure(figsize=(12, 8), dpi=120)
            # gs = gridspec.GridSpec(4, 3, figure=fig)
            # axs = [plt.subplot(gs[:2, 0]), plt.subplot(gs[2:, 0]), plt.subplot(gs[:2, 1]), plt.subplot(gs[2:, 1])]
            # if 'entropy' in df.columns:
            #     axs.extend([plt.subplot(gs[0:2, 2]), plt.subplot(gs[2:4, 2])])
            # else:
            #     axs.append(plt.subplot(gs[1:3, 2]))

            fig, axs = plt.subplots(2, 2, figsize=(8, 8),
                                    dpi=120)  # type:figure.Figure, List[axes.Axes]
        else:
            fig, axs = plt.subplots(1, 3, figsize=(12, 5),
                                    dpi=120)  # type:figure.Figure, List[axes.Axes]
        axs = np.array(axs).T.flatten()

        ax1 = axs[0]
        self.ax1 = ax1

        """ Subplot 1 - Top View """
        half_side = config.plane_side // 2
        bounds = [-half_side - 1, half_side + 1]
        ax1.set(title="Top view", xlim=bounds, ylim=bounds, autoscale_on=False, adjustable='box')
        major_ticks = np.arange(-half_side, half_side + 1, 5)
        minor_ticks = np.arange(-half_side, half_side + 1, 2.5)
        ax1.set_xticks(major_ticks)
        ax1.set_xticks(minor_ticks, minor=True)
        ax1.set_yticks(major_ticks)
        ax1.set_yticks(minor_ticks, minor=True)
        ax1.grid(which='both')
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.5)

        # Boundaries
        b = half_side + 0.1
        ax1.plot([-b, b, b, -b, -b], [-b, -b, b, b, -b], color='red', linewidth=1.2)

        # Thymio objects
        def thymio(color, fovline_cmap):
            """ Returns a dictionary of matplotlib objects for a Thymio instance on the top-down plot """
            thymio, = ax1.plot([], [], 'o', color=color, zorder=3, ms=1.)
            sensor_radius = ax1.add_artist(Circle((0, 0), 0, fill=False, linestyle='--', alpha=0.5, linewidth=0.7))
            square_map = ax1.add_artist(plt.Polygon([[0, 0] for _ in range(4)], closed=True, fill=None, edgecolor='r'))
            intent = ax1.add_line(Line2D([], [], linestyle='--', color='black', linewidth=1.0, alpha=0.8))
            lc1, lc2 = [ax1.add_collection(gradient_line(fovline_cmap)) for _ in range(2)]
            fov_area, clip_path = gradient_area(color)
            ax1.add_patch(clip_path)
            fov_area.set_clip_path(clip_path)
            return {'thymio': thymio, 'sensor_radius': sensor_radius, 'square_map': square_map, 'intent': intent,
                    'lc1': lc1, 'lc2': lc2, 'fov_area': fov_area, 'clip_path': clip_path}

        self.gt_artists = thymio('blue', pl.cm.Blues_r)
        self.odom_artists = thymio('orange', pl.cm.Oranges_r)

        # Targets
        # ax1.plot(static_targets[:, 0], static_targets[:, 1], 'ro', ms=2)

        # self.ax1_zoom = zoomed_inset_axes(ax1, 2, loc="upper left")
        circles = [plt.Circle((xi, yi), radius=0.0335, linewidth=0, edgecolor='none', facecolor='crimson', fill=True)
                   for xi, yi in static_targets]
        c = matplotlib.collections.PatchCollection(circles, match_original=True)
        self.ax1.add_collection(c)

        """ Subplot 2 - Camera Feed """

        ax2 = axs[1]
        ax2.set_title("Camera view")
        ax2.axis('off')
        self.camera = ax2.imshow(self.images[0])
        # self.sensor_text = ax2.text(0.5, 0.8, s=self.sensor_data[0], weight='bold', color='white',
        #                            ha="center", va="center", size=18, transform=ax2.transAxes)
        self.run_indicator = ax2.text(0.5, -0.1, '', ha="center", va="center", color='black', transform=ax2.transAxes,
                                      fontdict={'family': 'serif', 'size': 10})

        """ Subplot 3 - Occupancy Map """

        ax3 = axs[2]
        ax3.set_title('Occupancy map')
        ax3.axis('off')
        self.occ_map = ax3.imshow(self.colored_omaps[0])
        self.ax1.set_aspect('equal')
        fig.canvas.set_window_title("Simulation")
        if extra_info is not None:
            plt.suptitle(','.join([f'{k}: {v}' for k, v in extra_info.items()]))

        ratio = 20 / 0.8
        for r in scaled_full_robot_geometry(ratio):
            plot_transform(ax3, mktransf((10 - 0.5, 20 - 0.5, -np.pi / 2)) @ r, color='white',
                           length=config.max_sensing_distance * ratio)

        ax3.plot([10 / 20, 0], [0, 0.5], linewidth=.8, linestyle='--', color='blue', transform=ax3.transAxes)
        ax3.plot([10 / 20, 1], [0, 0.5], linewidth=.8, linestyle='--', color='blue', transform=ax3.transAxes)
        # ax3.add_artist(plt.Circle((0.5, 0.), radius=2/20, linewidth=0.8, linestyle='-', edgecolor='none', facecolor='crimson',
        #                          fill=False, transform=ax3.transAxes))
        """ Subplot 4 - Model prediction """

        self.omap_preds = None
        if 'predicted_map' in df.columns:
            self.omap_preds = df['predicted_map'].map(lambda x: np.array(x).reshape(20, 20)).values

            cmap = clr.LinearSegmentedColormap.from_list('diverging map',
                                                         [(0, (0, 1, 0)), (0.5, 'grey'), (1, (1, 0, 0))], N=256)
            self.omap_preds = [cmap(x) for x in self.omap_preds]
            ax4 = axs[3]
            ax4.set_title('Predicted occupancy map')
            ax4.axis('off')
            self.occ_map_pred = ax4.imshow(self.omap_preds[0])
            ax4.plot([10 / 20, 0], [0, 0.5], linewidth=1, linestyle='--', color='blue', transform=ax4.transAxes)
            ax4.plot([10 / 20, 1], [0, 0.5], linewidth=1, linestyle='--', color='blue', transform=ax4.transAxes)

            for r in scaled_full_robot_geometry(ratio):
                plot_transform(ax4, mktransf((10 - 0.5, 20 - 0.5, -np.pi / 2)) @ r, color='white',
                               length=config.max_sensing_distance * ratio)

        plt.tight_layout()
        self.anim = FuncAnimation(fig, self.animate, frames=frames, interval=1000. / rate, blit=True,
                                  init_func=self.init, repeat=True)

        if save_path is None:
            plt.show()
        else:
            plt.close()
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.anim.save(save_path, writer=writer,
                           progress_callback=lambda i, n: print(f'\rProcessing animation: {i * 100. / n:.2f} %',
                                                                end=''))
            print(f"\rProcess complete. Video file saved to: {save_path}")

    def init(self):
        # TODO: is this init_func really necessary
        """ FuncAnimation init function for the proper setup """
        for artist in [self.gt_artists, self.odom_artists]:
            artist['thymio'].set_data([], [])
            artist['intent'].set_data([], [])
            artist['lc1'].set_segments([])
            artist['lc2'].set_segments([])
            artist['fov_area'].set_transform(self.ax1.transData)
            artist['clip_path'].set_xy(np.empty((3, 2)))

        self.camera.set_data(self.images[0])
        # self.sensor_text.set_text(self.sensor_data[0])

        # self.occ_map.set_data(np.random.choice([0, 128, 255], size=config.occupancy_map_shape))
        self.occ_map.set_data(self.colored_omaps[0])

        artists = [*self.gt_artists.values(), *self.odom_artists.values(),
                   self.camera, self.run_indicator,  # self.sensor_text,
                   self.occ_map]  # *self.labels,
        if self.omap_preds is not None:
            self.occ_map_pred.set_data(self.omap_preds[0])
            artists.append(self.occ_map_pred)

        # for i, j in self.OCC_MAP_SQUARES:
        #    self.labels[i * config.occupancy_map_shape[0] + j].set_text('')

        return *artists,

    def animate(self, i):
        """ FuncAnimation animate function """

        for artist, positional_data in zip([self.odom_artists, self.gt_artists],
                                           [self.odom_pos_data, self.gt_pos_data]):
            x, y, theta = positional_data[i]

            artist['thymio'].set_data([x], [y])
            artist['intent'].set_data((x, x + cos(theta) * 0.8), (y, y + sin(theta) * 0.8))

            t = np.c_[[x, y], [x, y]].T + self.length * np.array([np.cos(theta + (d45 * np.array([-1, 1]))),
                                                                  np.sin(theta + (d45 * np.array([-1, 1])))]).T

            def segments(x, y, t):
                """ Returns a list of coordinates for the LineCollection's updated position """
                tx, ty = t
                rx = np.linspace(x, tx, self.fov_segments)
                ry = np.linspace(y, ty, self.fov_segments)
                points = np.array([rx, ry]).T.reshape(-1, 1, 2)
                return np.concatenate([points[:-1], points[1:]], axis=1)

            artist['lc1'].set_segments(segments(x, y, t[0]))
            artist['lc2'].set_segments(segments(x, y, t[1]))

            tr = transforms.Affine2D().rotate(theta + np.pi / 2).translate(x, y)
            artist['fov_area'].set_transform(tr + self.ax1.transData)
            artist['clip_path'].set_xy(np.array([[x, y], t[0], t[1]]))
            artist['fov_area'].set_clip_path(artist['clip_path'])

            artist['square_map'].set_xy([[x + cos(theta + i) * 0.4 * j, y + sin(theta + i) * 0.4 * j]
                                         for i, j in
                                         zip([-np.pi / 2, +np.pi / 2, +np.arctan(0.5), -np.arctan(0.5)],
                                             [1, 1, np.sqrt(5), np.sqrt(5)])])
            # artist['square_map']._update_patch_transform()
            # artist['square_map'].stale = True
            # artist['square_map'].set_transform(matplotlib.transforms.Affine2D().rotate_deg(theta*180/np.pi+90)
            #                                   + self.ax1.transData)

        self.ax1.set_xlim(x - 2, x + 2)
        self.ax1.set_ylim(y - 2, y + 2)

        d = np.cbrt(distance.euclidean(self.gt_pos_data[i], self.odom_pos_data[i]))
        for name, obj in self.odom_artists.items():
            if name not in ['sensor_radius', 'clip_path']:
                obj.set_alpha(min(1, max(0, d)))

        img = self.images[i]
        self.camera.set_data(img)
        # self.sensor_text.set_text(self.sensor_data[i])

        run = self.run_counter[i]
        end = self.run_dict['end'][run]
        length = self.run_dict['len'][run]
        self.run_indicator.set_text(f"Run {str(run).rjust(2)}: {str(length + i - end).rjust(4)}/{length} ")
        # f"({str(i).rjust(4)}/{len(self.images)})")

        # for k, j in self.OCC_MAP_SQUARES:
        #     cell = f'{self.omaps[i][k, j]:.0f}'
        #     self.labels[k * config.occupancy_map_shape[0] + j].set_text(cell)

        # else:
        #     if i != 0:
        #         for k, j in self.OCC_MAP_SQUARES:
        #             self.labels[k * config.occupancy_map_shape[0] + j].set_text('')

        artists = [*self.gt_artists.values(), *self.odom_artists.values(),
                   self.camera, self.run_indicator,  # self.sensor_text,
                   self.occ_map]  # *self.labels,

        self.occ_map.set_data(self.colored_omaps[i].astype('uint8'))
        if self.omap_preds is not None:
            self.occ_map_pred.set_data(self.omap_preds[i])
            artists.append(self.occ_map_pred)

        return *artists,


class TopDownAnimator:
    def __init__(self, df, rate=30, save_path=None, s=None, extra_info=None, weights_path=None, samples=None):
        # rc('axes', unicode_minus=False)
        # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        plt.close()
        self.rate = rate

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=rate, metadata=dict(artist='Me'), bitrate=1800)

        if s is not None:
            df = df.iloc[s]

        frames = len(df)

        # Cache

        def extract_images(_df, column):
            return _df[column] \
                .progress_apply(cv_from_binary) \
                .progress_apply(partial(Image.fromarray, mode="RGB")) \
                .to_numpy()

        self.front_feed = extract_images(df, 'head_camera_image_raw_image')
        self.topdown_feed = extract_images(df, 'image')

        self.run_counter = df['run'].to_numpy()
        rd = df.groupby('run')['x'].count().astype(int)
        rd = pd.DataFrame({'len': rd, 'end': rd.cumsum().astype(int)})
        self.run_dict = rd.to_dict()

        """ GridSpec Setup """

        fig = plt.figure(tight_layout=True, figsize=(10, 6))
        gs = gridspec.GridSpec(6, 6)

        ax_top = fig.add_subplot(gs[:3, :3])
        ax_front = fig.add_subplot(gs[:3, 3:])
        ax_pred = fig.add_subplot(gs[3:, :2])
        ax_entr = fig.add_subplot(gs[3:, 2:4])
        ax_mutu = fig.add_subplot(gs[3:, 4:])
        fig.align_labels()

        fig.canvas.set_window_title(f'Simulations: {", ".join(df["run"].unique().astype(str))}')
        if extra_info is not None:
            plt.suptitle(','.join([f'{k}: {v}' for k, v in extra_info.items()]))

        """ Subplot 1 - TopDown """

        ax_top.set_title("Top Down View")
        ax_top.axis('off')
        self.top = ax_top.imshow(self.topdown_feed[0])

        """ Subplot 2 - Camera Feed """

        ax_front.set_title("Camera view")
        ax_front.axis('off')
        self.front = ax_front.imshow(self.front_feed[0])

        """ Subplot 3 - Predicted Occupancy Map """

        self.omap_preds = np.full((len(df), 20, 20), fill_value=0.5)
        self.entr_preds = np.full((len(df), 20, 20), fill_value=0.5)
        self.mutu_preds = np.full((len(df), 20, 20), fill_value=0.5)

        ax_pred.set_title('Prediction', loc='center')
        ax_entr.set_title('Entropy', loc='center')
        ax_mutu.set_title('Mutual Information', loc='center')

        if 'predicted_map' in df or weights_path is not None:
            if weights_path is not None:
                print('using model to emulate high frequency predictions')
                bayes = samples is not None
                model, batch_size, device, _ = initialize_model(32, samples=samples, weights_path=weights_path)
                omap_preds, entr_maps, mi_maps = \
                    predictions_from_dataframe(model, df, batch_size, device, bayes,
                                               transform_function(bayes),
                                               target_column='head_camera_image_raw_image')
                self.entr_preds = entr_maps
                self.mutu_preds = mi_maps
            else:
                print('using low frequency registered predictions')
                omap_preds = df['predicted_map'].progress_apply(bytes_from_str).values
                omap_preds = [np.array(x).reshape(20, 20) for x in omap_preds]

            self.omap_preds = omap_preds

        ax_pred.axis('off')
        cmap = clr.LinearSegmentedColormap.from_list('diverging map', [(0, 'grey'), (0.5, 'red'), (1, 'yellow')], N=256)
        self.omap_preds = [cmap(x) for x in self.omap_preds]
        self.occ_map = ax_pred.imshow(self.omap_preds[0], cmap=cmap)
        self.entr_map = ax_entr.imshow(self.entr_preds[0], cmap=cmap)
        self.mutu_map = ax_mutu.imshow(self.mutu_preds[0], cmap=cmap)

        for ax, imsh in zip([ax_pred, ax_entr, ax_mutu], [self.occ_map, self.entr_map, self.mutu_map]):
            ax.set_aspect('equal')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.3)
            cbar = fig.colorbar(imsh, ax=ax, cax=cax)
            #cbar.set_ticks([0, 0.5, 1])
            #cbar.set_ticklabels([0, 0.5, 1])

        ratio = 20 / 0.8
        for r in scaled_full_robot_geometry(ratio):
            plot_transform(ax_pred, mktransf((10 - 0.5, 20 - 0.5, -np.pi / 2)) @ r, color='white',
                           length=config.max_sensing_distance * ratio)

        ax_pred.plot([10 / 20, 0], [0, 0.5], linewidth=.8, linestyle='--', color='white', transform=ax_pred.transAxes)
        ax_pred.plot([10 / 20, 1], [0, 0.5], linewidth=.8, linestyle='--', color='white', transform=ax_pred.transAxes)

        """ Final Steps for animation and saving """

        self.anim = FuncAnimation(fig, self.animate, frames=frames, interval=1000. / rate, blit=True,
                                  init_func=self.init, repeat=True)

        self.save_path = save_path
        if save_path is None:
            plt.show()  # for debugging
        else:
            plt.close()
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.anim.save(save_path, writer=writer,
                           progress_callback=lambda i, n: print(f'\rProcessing animation: {i * 100. / n:.2f} %',
                                                                end=''))
            print(f"\rAnimation complete. Video file saved to: {save_path}")

    def init(self):
        self.top.set_data(self.topdown_feed[0])
        self.front.set_data(self.front_feed[0])
        self.occ_map.set_data(self.omap_preds[0])
        self.entr_map.set_data(self.entr_preds[0])
        self.mutu_map.set_data(self.mutu_preds[0])
        return self.top, self.front, self.occ_map,self.entr_map,self.mutu_map,

    def animate(self, i):
        self.top.set_data(self.topdown_feed[i])
        self.front.set_data(self.front_feed[i])
        self.occ_map.set_data(self.omap_preds[i])
        self.entr_map.set_data(self.entr_preds[i])
        self.mutu_map.set_data(self.mutu_preds[i])
        return self.top, self.front, self.occ_map,self.entr_map,self.mutu_map,
