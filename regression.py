class Regression: # Inherits from analytics object

    def __init__(self, X, y, z=None, X_test = None, y_test = None):
        """Initialize the Regression Class; inherits from Analysis Class."""
        self.X = X
        self.y = y 
        self.z = z
        self.X_test = X_test
        self.y_test = y_test

    def linear(self, se_type = 'standard', 
               cluster = None, 
               include_intercept = True): 
        
        import seaborn as sns 
        import matplotlib.pyplot as plt
        import numpy as np 
        import pandas as pd 
        from scipy.stats import t as t_dist
            
        # Add intercept
        N = (self.X).shape[0]
        if include_intercept == True:
            self.X['Intercept'] = np.ones(N)
        K = self.X.shape[1]

        # Run regression
        XpX = self.X.T @ self.X # Compute X'Z
        Xpy = self.X.T @ self.y # Compute X'y
        beta_hat = np.linalg.solve(XpX, Xpy) # Solve normal equations
        #
        y_hat = self.X @ beta_hat # Compute predictions
        residuals = self.y-y_hat # Compute residuals
        #
        SSE =  np.inner(residuals,residuals) # Compute SSE
        SER = np.sum(SSE)/(N-K) # Compute standard error of regression
        rsq = 1 - SSE/np.inner( self.y-np.mean(self.y),self.y-np.mean(self.y)) # Compute Rsq
        #
        if se_type == 'standard':
            V = SER * np.linalg.inv(XpX)
            SE = np.sqrt(np.diag(V))
        elif se_type == 'robust':
            e_sq = residuals**2
            bread = np.linalg.inv(XpX) 
            meat = self.X.T @ np.diag(e_sq) @ self.X
            V = (N/(N-K)) * bread @ meat @ bread
            SE = np.sqrt(np.diag(V))
            
        t_stat = beta_hat/SE
        pval = 2*(t_dist.cdf(-abs(t_stat), N-K))
        nstars = np.zeros(len(pval))
        nstars[ pval <= .01 ] = 3
        nstars[ (pval > .01) * (pval <= .05) ] = 2
        nstars[ (pval > .05) * (pval <= .10) ] = 1
        stars = ['*'*int(i) for i in nstars]
        
        output = pd.DataFrame({'Variable':(self.X).columns,
                            'Coefficient':beta_hat,
                            'Std.Err':SE,
                            't Statistic':t_stat,
                            'p-Value': pval,
                            'Significance': stars
                            })

        return({'beta_hat':beta_hat,
                'y_hat':y_hat,
                'residuals':residuals,
                'rsq':rsq,
                'SSE':SSE,
                'SER':SER,
                'SE':SE,
                'table':output})
