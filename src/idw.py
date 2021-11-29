"""
Definition and implementation of the Inverse Distance Weighting Interpolation

One must add the following entry to PYTHONPATH:
    export PYTHONPATH="/mnt/0A853CB226EABEA6/IC/IC_ENZO/implementacao/:$PYTHONPATH"
    source .bashrc ...
"""

class InverseDistanceWeighting(object):


    def __init__(self):
        self._all_points = None


    def add_point(self, value: float, dist: float, error = None, alias = None):
        """
        Parameters
        ----------
        `loc`: tuple
            Tupla contento as valores (latitude, longitude) - ambos em base decimal

        `value`: float
            Prediction of the reference point

        `error`: float
            Estimator error ((RMSE+MAE) / 2)
        """

        if self._all_points == None:
            self._all_points = []

        self._all_points.append({
            'value': value,
            'dist': dist,
            'error': error,
            'alias': alias,
        })


    def total_points(self):
        return 0 if self._all_points == None else len(self._all_points)


    def dispose_points(self):
        self._all_points = None


    def w_metric(self, ki, p, method):

        if method == 'standard':
            return 1 / (ki['dist']**p)

        elif method == 'custom':
            return (1 / (ki['dist']**p)) * (1 / ki['error'])


    def interpolate(self, p = 2, method = 'standard'):
        """
        Calculates the prediction value of `p0` for a given set of points

        Parameters
        ------------
        p : float, default=2
            Smoothing factor applied to the distance (d).

        method : str, default='standard'
            Interpolation method to be used:
                - standard: standard interpolation considering only the distance `d`
                - custom: interpolation considering both distance and error of the
                estimators

        Returns
        ---------
            Weighted sum based on the inverse distance of point `p0` to the points
            in the object.
        """

        if self._all_points == None:
            raise AttributeError('Reference point have not been initialized.')

        upper_sum = 0
        lower_sum = 0

        for k in self._all_points:
            wi = self.w_metric(ki = k, p = p, method = method)

            upper_sum += wi * k['value']
            lower_sum += wi

        return upper_sum / lower_sum
