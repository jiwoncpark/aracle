import numpy as np
import matplotlib.pyplot as plt

class ToySquares:
    """A set of squares that grow and shift to the right over time

    Parameters
    ----------
    canvas_size : int
        size of the canvas on which the toy squares fall, in pixels
    n_objects : int
        number of toy squares to spawn

    """
    def __init__(self, canvas_size, n_objects):
        self.canvas_size = canvas_size
        self.n_objects = n_objects
        self.initialize_positions()
        self.initialize_sizes()
        self.set_growth_rates()
        self.rightward_shift = 2 # pixels

    def initialize_positions(self):
        """Initialize the initial positions of the squares, with respect to the lower left of the square

        """
        # Initialize x on the left half, so it doesn't fall out of bounds too quickly as it moves rightward across the canvas
        self.x_pos = (np.random.rand(self.n_objects)*self.canvas_size*0.5).astype(int)
        self.y_pos = (np.random.rand(self.n_objects)*self.canvas_size).astype(int)
        self.in_the_canvas = np.ones(self.n_objects).astype(bool)

    def initialize_sizes(self):
        """Initialize the initial sizes of the squares, as the number of pixels per edge

        """
        allowed_sizes = np.arange(1, 5)
        prob = np.ones(len(allowed_sizes))
        prob /= np.sum(prob)
        sizes = np.random.choice(allowed_sizes, size=self.n_objects, p=prob, replace=True)
        self.x_sizes = sizes
        self.y_sizes = sizes

    def set_growth_rates(self):
        """Randomly set the size increase that is applied every time step

        """ 
        allowed_growth_rates = np.arange(1, 3)
        prob = np.ones(len(allowed_growth_rates))
        prob /= np.sum(prob)
        self.growth_rates = np.random.choice(allowed_growth_rates, size=self.n_objects, p=prob, replace=True)

    def increment_time_step(self):
        """Advance one time step, updating object properties accordingly

        """
        self.grow()
        self.shift_right()
        self.update_in_canvas()

    def grow(self):
        """Grow the sizes of the objects by their respective growth rates

        """
        self.x_sizes += self.growth_rates
        self.y_sizes += self.growth_rates

    def shift_right(self):
        """Shift the objects to the right by two pixels

        """
        self.x_pos += self.rightward_shift

    def update_in_canvas(self):
        """Evaluate whether the objects fall within the canvas and, if they get truncated by the canvas bounds, what the effective sizes are

        """
        self.x_sizes = np.minimum(self.x_sizes, self.canvas_size - self.x_pos)
        self.y_sizes = np.minimum(self.x_sizes, self.canvas_size - self.y_pos)
        x_in_canvas = (self.x_sizes > 0.0)
        y_in_canvas = (self.y_sizes > 0.0)
        self.in_canvas = np.logical_and(x_in_canvas, y_in_canvas)

    def export_image(self, img_path):
        """Export the current object states to disk as an npy file

        Paramters
        ---------
        img_path : str or os.path object
            path of image file to be saved

        """
        canvas = np.zeros((self.canvas_size, self.canvas_size))
        for obj in range(self.n_objects):
            canvas[self.x_pos[obj]:self.x_pos[obj] + self.x_sizes[obj],
            self.y_pos[obj]:self.y_pos[obj] + self.y_sizes[obj]] = 1.0
        np.save(img_path, canvas.T) # transpose b/c numpy indexing conventions

if __name__ == '__main__':
    toy_squares = ToySquares(canvas_size=224, n_objects=3)
    toy_squares.increment_time_step()