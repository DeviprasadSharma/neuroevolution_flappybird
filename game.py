#! /usr/bin/env python3

"""Flappy Bird, implemented using Pygame."""

import math
import os
from random import randint
from collections import deque
import numpy as np
import pygame
from pygame.locals import *
import neuro_evolution

FPS = 60
ANIMATION_SPEED = 0.18  # pixels per millisecond
WIN_WIDTH = 284 * 2     # BG image size: 284x512 px; tiled twice
WIN_HEIGHT = 512


class Bird(pygame.sprite.Sprite):
    """Represents the bird controlled by the player.

    The bird is the 'hero' of this game.  The player can make it climb
    (ascend quickly), otherwise it sinks (descends more slowly).  It must
    pass through the space in between pipes (for every pipe passed, one
    point is scored); if it crashes into a pipe, the game ends.

    Attributes:
    x: The bird's X coordinate.
    y: The bird's Y coordinate.
    msec_to_climb: The number of milliseconds left to climb, where a
        complete climb lasts Bird.CLIMB_DURATION milliseconds.

    Constants:
    WIDTH: The width, in pixels, of the bird's image.
    HEIGHT: The height, in pixels, of the bird's image.
    SINK_SPEED: With which speed, in pixels per millisecond, the bird
        descends in one second while not climbing.
    CLIMB_SPEED: With which speed, in pixels per millisecond, the bird
        ascends in one second while climbing, on average.  See also the
        Bird.update docstring.
    CLIMB_DURATION: The number of milliseconds it takes the bird to
        execute a complete climb.
    """

    WIDTH = HEIGHT = 32
    SINK_SPEED = 0.18
    CLIMB_SPEED = 0.3
    CLIMB_DURATION = 333.3
    JUMP = -6

    def __init__(self, x, y, msec_to_climb, images):
        """Initialise a new Bird instance.

        Arguments:
        x: The bird's initial X coordinate.
        y: The bird's initial Y coordinate.
        msec_to_climb: The number of milliseconds left to climb, where a
            complete climb lasts Bird.CLIMB_DURATION milliseconds.  Use
            this if you want the bird to make a (small?) climb at the
            very beginning of the game.
        images: A tuple containing the images used by this bird.  It
            must contain the following images, in the following order:
                0. image of the bird with its wing pointing upward
                1. image of the bird with its wing pointing downward
        """
        super(Bird, self).__init__()
        self.x, self.y = x, y
        self.msec_to_climb = msec_to_climb
        self._img_wingup, self._img_wingdown = images
        self._mask_wingup = pygame.mask.from_surface(self._img_wingup)
        self._mask_wingdown = pygame.mask.from_surface(self._img_wingdown)
        self.alive = True
        self.gravity = 0

    def flap(self):
        self.gravity = Bird.JUMP

    def update(self):
        """Update the bird's position.

        This function uses the cosine function to achieve a smooth climb:
        In the first and last few frames, the bird climbs very little, in the
        middle of the climb, it climbs a lot.
        One complete climb lasts CLIMB_DURATION milliseconds, during which
        the bird ascends with an average speed of CLIMB_SPEED px/ms.
        This Bird's msec_to_climb attribute will automatically be
        decreased accordingly if it was > 0 when this method was called.

        Arguments:
        delta_frames: The number of frames elapsed since this method was
            last called.
        """
        self.gravity += Bird.CLIMB_SPEED
        self.y += self.gravity

    @property
    def image(self):
        """Get a Surface containing this bird's image.

        This will decide whether to return an image where the bird's
        visible wing is pointing upward or where it is pointing downward
        based on pygame.time.get_ticks().  This will animate the flapping
        bird, even though pygame doesn't support animated GIFs.
        """
        if pygame.time.get_ticks() % 500 >= 250:
            return self._img_wingup
        else:
            return self._img_wingdown

    @property
    def mask(self):
        """Get a bitmask for use in collision detection.

        The bitmask excludes all pixels in self.image with a
        transparency greater than 127."""
        if pygame.time.get_ticks() % 500 >= 250:
            return self._mask_wingup
        else:
            return self._mask_wingdown

    @property
    def rect(self):
        """Get the bird's position, width, and height, as a pygame.Rect."""
        return Rect(self.x, self.y, Bird.WIDTH, Bird.HEIGHT)


class PipePair(pygame.sprite.Sprite):
    """Represents an obstacle.

    A PipePair has a top and a bottom pipe, and only between them can
    the bird pass -- if it collides with either part, the game is over.

    Attributes:
    x: The PipePair's X position.  This is a float, to make movement
        smoother.  Note that there is no y attribute, as it will only
        ever be 0.
    image: A pygame.Surface which can be blitted to the display surface
        to display the PipePair.
    mask: A bitmask which excludes all pixels in self.image with a
        transparency greater than 127.  This can be used for collision
        detection.
    top_pieces: The number of pieces, including the end piece, in the
        top pipe.
    bottom_pieces: The number of pieces, including the end piece, in
        the bottom pipe.

    Constants:
    WIDTH: The width, in pixels, of a pipe piece.  Because a pipe is
        only one piece wide, this is also the width of a PipePair's
        image.
    PIECE_HEIGHT: The height, in pixels, of a pipe piece.
    ADD_INTERVAL: The interval, in milliseconds, in between adding new
        pipes.
    """

    WIDTH = 80
    PIECE_HEIGHT = 32
    ADD_INTERVAL = 3000

    def __init__(self, pipe_end_img, pipe_body_img):
        """Initialises a new random PipePair.

        The new PipePair will automatically be assigned an x attribute of
        float(WIN_WIDTH - 1).

        Arguments:
        pipe_end_img: The image to use to represent a pipe's end piece.
        pipe_body_img: The image to use to represent one horizontal slice
            of a pipe's body.
        """
        self.x = float(WIN_WIDTH - 1)
        self.score_counted = False

        self.image = pygame.Surface((PipePair.WIDTH, WIN_HEIGHT), SRCALPHA)
        self.image.convert()   # speeds up blitting
        self.image.fill((0, 0, 0, 0))
        total_pipe_body_pieces = int(
            (WIN_HEIGHT -                  # fill window from top to bottom
             3 * Bird.HEIGHT -             # make room for bird to fit through
             3 * PipePair.PIECE_HEIGHT) /  # 2 end pieces + 1 body piece
            PipePair.PIECE_HEIGHT          # to get number of pipe pieces
        )
        self.bottom_pieces = randint(1, total_pipe_body_pieces)
        self.top_pieces = total_pipe_body_pieces - self.bottom_pieces

        # bottom pipe
        for i in range(1, self.bottom_pieces + 1):
            piece_pos = (0, WIN_HEIGHT - i*PipePair.PIECE_HEIGHT)
            self.image.blit(pipe_body_img, piece_pos)
        bottom_pipe_end_y = WIN_HEIGHT - self.bottom_height_px
        bottom_end_piece_pos = (0, bottom_pipe_end_y - PipePair.PIECE_HEIGHT)
        self.image.blit(pipe_end_img, bottom_end_piece_pos)

        # top pipe
        for i in range(self.top_pieces):
            self.image.blit(pipe_body_img, (0, i * PipePair.PIECE_HEIGHT))
        top_pipe_end_y = self.top_height_px
        self.image.blit(pipe_end_img, (0, top_pipe_end_y))

        # compensate for added end pieces
        self.top_pieces += 1
        self.bottom_pieces += 1

        # for collision detection
        self.mask = pygame.mask.from_surface(self.image)

    @property
    def top_height_px(self):
        """Get the top pipe's height, in pixels."""
        return self.top_pieces * PipePair.PIECE_HEIGHT

    @property
    def bottom_height_px(self):
        """Get the bottom pipe's height, in pixels."""
        return self.bottom_pieces * PipePair.PIECE_HEIGHT

    @property
    def visible(self):
        """Get whether this PipePair on screen, visible to the player."""
        return -PipePair.WIDTH < self.x < WIN_WIDTH

    @property
    def rect(self):
        """Get the Rect which contains this PipePair."""
        return Rect(self.x, 0, PipePair.WIDTH, PipePair.PIECE_HEIGHT)

    def update(self, delta_frames=1):
        """Update the PipePair's position.

        Arguments:
        delta_frames: The number of frames elapsed since this method was
            last called.
        """
        self.x -= ANIMATION_SPEED * frames_to_msec(delta_frames)

    def collides_with(self, bird):
        """Get whether the bird collides with a pipe in this PipePair.

        Arguments:
        bird: The Bird which should be tested for collision with this
            PipePair.
        """
        return pygame.sprite.collide_mask(self, bird)


def load_images():
    """Load all images required by the game and return a dict of them.

    The returned dict has the following keys:
    background: The game's background image.
    bird-wingup: An image of the bird with its wing pointing upward.
        Use this and bird-wingdown to create a flapping bird.
    bird-wingdown: An image of the bird with its wing pointing downward.
        Use this and bird-wingup to create a flapping bird.
    pipe-end: An image of a pipe's end piece (the slightly wider bit).
        Use this and pipe-body to make pipes.
    pipe-body: An image of a slice of a pipe's body.  Use this and
        pipe-body to make pipes.
    """

    def load_image(img_file_name):
        """Return the loaded pygame image with the specified file name.

        This function looks for images in the game's images folder
        (./images/).  All images are converted before being returned to
        speed up blitting.

        Arguments:
        img_file_name: The file name (including its extension, e.g.
            '.png') of the required image, without a file path.
        """
        file_name = os.path.join('.', 'images', img_file_name)
        img = pygame.image.load(file_name)
        img.convert()
        return img

    return {'background': load_image('background.png'),
            'pipe-end': load_image('pipe_end.png'),
            'pipe-body': load_image('pipe_body.png'),
            # images for animating the flapping bird -- animated GIFs are
            # not supported in pygame
            'bird-wingup': load_image('bird_wing_up.png'),
            'bird-wingdown': load_image('bird_wing_down.png'),
            'test': load_image('test.png')}


def frames_to_msec(frames, fps=FPS):
    """Convert frames to milliseconds at the specified framerate.

    Arguments:
    frames: How many frames to convert to milliseconds.
    fps: The framerate to use for conversion.  Default: FPS.
    """
    return 1000.0 * frames / fps


def msec_to_frames(milliseconds, fps=FPS):
    """Convert milliseconds to frames at the specified framerate.

    Arguments:
    milliseconds: How many milliseconds to convert to frames.
    fps: The framerate to use for conversion.  Default: FPS.
    """
    return fps * milliseconds / 1000.0

class Game():
    BIRD_X = int(WIN_HEIGHT/2 - Bird.HEIGHT/2)
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption('Pygame Flappy Bird')

        self.clock = pygame.time.Clock()
        self.score_font = pygame.font.SysFont(None, 32, bold=True)  # default font
        self.images = load_images()
        # set ai
        self.ai = neuro_evolution.NeuroEvolution()
        self.generation = 0
        # set pipes
        self.pipes = deque()
        self.record = []
        self.logpath = 'scores.txt'
        self.maxscore = 0

    def start(self):
        self.score = 0
        self.pipes = deque()
        self.birds = []

        self.gen = self.ai.next_generation() #nn s
        for i in range(len(self.gen)):
            bird = Bird(50, Game.BIRD_X, 2,
                (self.images['bird-wingup'], self.images['bird-wingdown']))
            self.birds.append(bird)

        self.generation += 1
        self.alives = len(self.birds)

    def get_input(self,bird,size = 3):
        input = np.zeros(size)
        input[0] = bird.y
        # only the first pipe are considered
        #n = (size - 1)/3 - 1
        #for i,pp in enumerate(self.pipes):
        #    input[i*3+1] = pp.x
        #    input[i*3+2] = pp.bottom_height_px / WIN_HEIGHT
        #    input[i*3+3] = pp.top_height_px / WIN_HEIGHT
        #    if i == n:
        #        break
        for pp in self.pipes:
            if pp.x + PipePair.WIDTH > bird.x:
                input[1] = pp.top_height_px
                input[2] = pp.x + PipePair.WIDTH
                break
        #if len(self.pipes) > 0 and bird.y < self.pipes[0].bottom_height_px:
        #    input[-1] = -1.0
        #else:
        #    input[-1] = 1.0
        #input[-1] = -1.0
        return input

    @property
    def all_dead(self):
        for bird in self.birds:
            if bird.alive:
                return False
        return True

    def update(self, paused = False):
        # generate pipes
        if not (paused or self.frame_clock % msec_to_frames(PipePair.ADD_INTERVAL)):
            pp = PipePair(self.images['pipe-end'], self.images['pipe-body'])
            self.pipes.append(pp)
        # generate background
        for x in (0, WIN_WIDTH / 2):
            self.screen.blit(self.images['background'], (x, 0))
        # update pipes
        while self.pipes and not self.pipes[0].visible:
            self.pipes.popleft()

        for p in self.pipes:
            p.update()
            self.screen.blit(p.image, p.rect)
        #update score
        for p in self.pipes:
            if p.x + PipePair.WIDTH < Game.BIRD_X and not p.score_counted:
                self.score += 1
                p.score_counted = True
        self.score += 0.5
        # generate bird
        for i,bird in enumerate(self.birds):
            if bird.alive:
                inputs = self.get_input(bird)
                res = self.gen[i].feedforward(inputs)
                #print(res)
                if res[0] > 0.5: # flap
                    bird.flap()
                bird.update()
                self.screen.blit(bird.image, bird.rect)

                pipe_collision = any(p.collides_with(bird) for p in self.pipes)
                if pipe_collision or 0 >= bird.y or bird.y >= WIN_HEIGHT - Bird.HEIGHT:
                    bird.alive = False
                    self.alives -= 1
                    self.ai.network_score(self.score,self.gen[i])
                    if self.all_dead:
                        self.frame_clock = -1
                        self.record.append(self.score)
                        if self.score > self.maxscore:
                            self.maxscore = self.score
                        if self.generation % 100 == 0: #output every 100 generations
                            with open(self.logpath,'a') as f:
                                for sc in self.record:
                                    f.write(str(sc)+',')
                            self.record = []
                        #self.ai.output() #record the score all ais
                        self.start()
        # print score
        score_surface = self.score_font.render('score:{0:.1f}'.format(self.score), True, (255, 255, 255))
        maxscore_surface = self.score_font.render('max:{0:.1f}'.format(self.maxscore), True, (255, 255, 255))
        generation_surface = self.score_font.render('generation:{}'.format(self.generation), True, (255, 255, 255))
        alive_surface = self.score_font.render('alive:{}/50'.format(self.alives), True, (255, 255, 255))
        score_x = WIN_WIDTH/2 - score_surface.get_width()/2
        self.screen.blit(maxscore_surface, (score_x, 0))
        self.screen.blit(score_surface, (score_x, maxscore_surface.get_height()))
        self.screen.blit(generation_surface, (score_x, maxscore_surface.get_height()*2))
        self.screen.blit(alive_surface, (score_x, score_surface.get_height()*3))
        pygame.display.flip()
        self.frame_clock += 1

    def run(self, FPS=FPS):
        done = paused = False
        self.frame_clock = 0
        self.start()
        while not done:
            self.clock.tick(FPS)
            for e in pygame.event.get():
                if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                    done = True
                    break
                elif e.type == KEYUP and e.key in (K_PAUSE, K_p):
                    paused = not paused
            if paused:
                continue
            #self.screen.fill(BACKGROUND)
            self.update(paused)
        print('Over')
        pygame.quit()
        with open('lastgeneration.txt','w') as f:
            for x in self.ai.gene.generations[-1].individuals:
                f.write(str(x.netweights))

    def debug(self):
        """ If you want to simply play it by yourself"""

        pygame.init()

    # the bird stays in the same x position, so bird.x is a constant
    # center bird on screen
        bird = Bird(50, int(WIN_HEIGHT/2 - Bird.HEIGHT/2), 2,
                (self.images['bird-wingup'], self.images['bird-wingdown']))

        self.pipes = deque()

        frame_clock = 0  # this counter is only incremented if the game isn't paused
        self.score = 0
        done = paused = False
        while not done:
            self.clock.tick(FPS)

        # Handle this 'manually'.  If we used pygame.time.set_timer(),
        # pipe addition would be messed up when paused.
            if not (paused or frame_clock % msec_to_frames(PipePair.ADD_INTERVAL)):
                pp = PipePair(self.images['pipe-end'], self.images['pipe-body'])
                self.pipes.append(pp)

            for e in pygame.event.get():
                if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                    done = True
                    break
                elif e.type == KEYUP and e.key in (K_PAUSE, K_p):
                    paused = not paused
                elif e.type == MOUSEBUTTONUP or (e.type == KEYUP and
                    e.key in (K_UP, K_RETURN, K_SPACE)):
                    bird.flap()

            if paused:
                continue  # don't draw anything

        # check for collisions
            pipe_collision = any(p.collides_with(bird) for p in self.pipes)
            if pipe_collision or 0 >= bird.y or bird.y >= WIN_HEIGHT - Bird.HEIGHT:
                done = True

            for x in (0, WIN_WIDTH / 2):
                self.screen.blit(self.images['background'], (x, 0))

            while self.pipes and not self.pipes[0].visible:
                self.pipes.popleft()

            for p in self.pipes:
                p.update()
                self.screen.blit(p.image, p.rect)

            bird.update()
            print(bird.msec_to_climb)
            self.screen.blit(bird.image, bird.rect)

        # update and display score
            for p in self.pipes:
                if p.x + PipePair.WIDTH < bird.x and not p.score_counted:
                    self.score += 1
                    p.score_counted = True

            score_surface = self.score_font.render(str(self.score), True, (255, 255, 255))
            score_x = WIN_WIDTH/2 - score_surface.get_width()/2
            self.screen.blit(score_surface, (score_x, PipePair.PIECE_HEIGHT))

            pygame.display.flip()
            frame_clock += 1
        print('Game over! Score: %i' % score)
        pygame.quit()


def main():
    game = Game()
    game.run()
    #print('generation')
    #print(game.record['generation'])
    #print('highest_score')
    #print(game.record)
    #game.debug()



if __name__ == '__main__':
    # If this module had been imported, __name__ would be 'flappybird'.
    # It was executed (e.g. by double-clicking the file), so call main.
    main()
