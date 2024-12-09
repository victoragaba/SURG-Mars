import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt

# inv_model.plot_iterates_3D(s=25, optimal=True, index=2)

class InversionModel:
    def plot_iterates_3D(self, elev=30, azim=45, cmap='rainbow', s=10, optimal=False, index=2): 
        '''
        Make a 3D scatter plot of the iterates (psi, delta, lambda) and project
        them onto the 2D planes: psi-delta, psi-lambda, delta-lambda.
        Color the points by iteration number.
        
        Args:
            optimal (bool): If True, plot only the optimal points.
            warm (bool): If False, 
        '''
        if index == 0 or index == 1:
            if not optimal:
                self.mirror(['optimals', 'iterates'], index)
                iterates = self.iterates
                optimal_iterates = self.optimal_iterates
            else:
                self.mirror(['optimals'], index)
                optimal_iterates = self.optimal_iterates
        elif index == 2:
            iterates, optimal_iterates = [], []
            if not optimal:
                self.mirror(['optimals', 'iterates'], 0)
                iterates.extend(self.iterates)
                optimal_iterates.extend(self.optimal_iterates)
                self.mirror(['optimals', 'iterates'], 1)
                iterates.extend(self.iterates)
                optimal_iterates.extend(self.optimal_iterates)
            else:
                self.mirror(['optimals'], 0)
                optimal_iterates.extend(self.optimal_iterates)
                self.mirror(['optimals'], 1)
                optimal_iterates.extend(self.optimal_iterates)
        
        # convert the angles to degrees
        if not optimal:
            strikes = [np.rad2deg(m[0]) for m in iterates]
            dips = [np.rad2deg(m[1]) for m in iterates]
            rakes = [np.rad2deg(m[2]) for m in iterates]
            weights = -np.array(self.misfits)
            if index == 2: weights = np.concatenate([weights, weights])
            
            # normalize weights for consistent coloring
            norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
            cmap_instance = plt.cm.get_cmap(cmap)
            
        opt_strikes = [np.rad2deg(m[0]) for m in optimal_iterates]
        opt_dips = [np.rad2deg(m[1]) for m in optimal_iterates]
        opt_rakes = [np.rad2deg(m[2]) for m in optimal_iterates]
        
        
        # create a 3D scatter plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        if not optimal:
            scatter = ax.scatter(
                strikes, dips, rakes,
                c=weights, cmap=cmap_instance, norm=norm, s=s
            )
        ax.scatter(
            opt_strikes, opt_dips, opt_rakes,
            c='black', marker='*', s=s, label='Optimal'
        )
        ax.set_xlabel('Strike (deg)')
        ax.set_ylabel('Dip (deg)')
        ax.set_zlabel('Rake (deg)')
        plt.title('3D visualization of the iterates', fontsize=15)
        
        # adjust the view angle
        ax.view_init(elev=elev, azim=azim)
        
        # add a colorbar to the figure
        if not optimal:
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
            cbar.set_label('Cosine similarity')
            ax.legend()
        
        plt.show()
        
        