#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

class Lab:
    def __init__(self, **kwargs):
        # Initialize class variables for coefficients and their uncertainties
        self.a  = 0  # Coefficient a
        self.b  = 0  # Coefficient b
        self.sa = 0  # Uncertainty in a
        self.sb = 0  # Uncertainty in b 
        
        # Retrieve file names from keyword arguments
        m_file = kwargs.get('file', None)
        m_filex = kwargs.get('file_x', None)
        m_filey = kwargs.get('file_y', None)
        
        if m_file:
            # If a single file is provided, read data from it
            self.m_data = self.readf(m_file)
        else:
            # If separate x and y files are provided, read data from both
            datax = self.readf(m_filex)
            datay = self.readf(m_filey)
            self.m_datax = self.avg(datax)  # Calculate average for x data
            self.m_datay = self.avg(datay)  # Calculate average for y data
            
            # Combine the averaged results for printing
            self.m_data = np.hstack((self.m_datax, self.m_datay))
        print(self.m_data)  # Print the combined data
        
        # Save the averaged data to a text file
        np.savetxt('avg.txt', self.m_data, header='Mean\tStd Dev', delimiter='\t', fmt='%.6f')

  
    def readf(self, file):
        # Read data from a specified file and handle errors
        try:
            return np.loadtxt(file, dtype=float, delimiter='\t', comments='#')
        except FileNotFoundError:
            print(f"Error: The file '{file}' was not found.")
            return None
        except ValueError as e:
            print(f"Error reading the file: {e}")
            return None


    def __str__(self):
        # String representation of the class, returns the data
        return f'{self.m_data}'


    def avg(self, dat):
        # Calculate the average and standard deviation for each row in the data
        size = dat.shape
        ans = np.zeros((size[0], 2), dtype=float)  # Create array for mean and std deviation
        
        for i in range(size[0]):
            row_sum = np.sum(dat[i])  # Sum values in the row for mean calculation
            ans[i][0] = row_sum / size[1]  # Calculate mean
            
            # Calculate standard deviation
            variance = np.sum((dat[i] - ans[i][0]) ** 2) / (size[1] - 1 if size[1] < 30 else size[1])
            ans[i][1] = np.sqrt(variance)  # Standard deviation

        return ans 
    
    def least_square(self, check=False, dy_dx=0):
        # Perform least squares fitting on the data
        data = self.m_data
        n = data.shape[0]  # Number of data points
        x, y, xy, x2, w, sum_err, syn = [], [], [], [], [], 0, []  # Initialize lists and variables
        
        for i in range(n):
            weight = 1  # Default weight
            if check:
                # Calculate the weighted error if checking for uncertainties
                syn.append(np.sqrt(data[i][3]**2 + (dy_dx * data[i][1])**2))
                if syn[i] == 0:
                    raise ValueError("The new y error is zero check data")
                data[i][1] = 0  # Set the error to zero for this calculation
                data[i][3] = syn[i]  # Update the y error
                weight = 1 / (syn[i]**2)  # Weight based on uncertainty
                w.append(weight)
                
            # Collect weighted x and y values
            x.append(data[i][0] * weight)
            y.append(data[i][2] * weight)
            xy.append(data[i][0] * data[i][2] * weight)
            x2.append(weight * data[i][0]**2)
        
        sw = sum(w)  # Total weight
        sx = sum(x)  # Weighted sum of x
        sy = sum(y)  # Weighted sum of y
        sxy = sum(xy)  # Weighted sum of xy
        sx2 = sum(x2)  # Weighted sum of x^2

        if check:
            # Calculate coefficients and their uncertainties when checking
            d = sw * sx2 - sx**2
            b = (sx2 * sy - sx * sxy) / d
            a = (sw * sxy - sx * sy) / d
            sa = np.sqrt(sw / d)
            sb = np.sqrt(sx2 / d)
        else:
            # Regular fitting process
            if n <= 2:
                raise ValueError("number of data must be more than 2")
            d = n * sx2 - sx**2
            a = (n * sxy - sx * sy) / d
            b = (sx2 * sy - sx * sxy) / d  
            
            # Calculate the sum of squared errors
            for i in range(n):
                sum_err += (data[i][2] - b - a * data[i][0])**2
            
            er_y = np.sqrt(sum_err / (n - 2))  # Estimate of the standard error
            print(er_y)  # Print the standard error
            
            # Calculate uncertainties in coefficients
            sa = np.sqrt((n / d) * er_y**2)
            sb = np.sqrt((sx2 / d) * er_y**2)
            print(sa, sb)  # Print uncertainties

        # Store coefficients and uncertainties
        self.a = a
        self.b = b
        self.sa = sa
        self.sb = sb
        
        # Determine file name based on whether checking or not
        file_name = "least_square_syn.txt" if check else "least_square.txt"

        with open(file_name, "w") as file:
            # Write the results to the file
            file.write(f"a = {a:.10f}\n")
            file.write(f"b = {b:.10f}\n")
            file.write(f"sa = {sa:.10f}\n")
            file.write(f"sb = {sb:.10f}\n")
            file.write(f"d = {d:.10f}\n")
            
            # Write header for data table
            file.write("\nData Table:\n")
            header = f"{'x':>20} {'y':>20} {'x2':>20} {'xy':>20}"
            if check:
                header += f"{' syn':>20} {' w':>20}"
            file.write(header + "\n")
            file.write("=" * 150 + "\n")  # Separator for clarity
            
            # Write each row of the processed data
            for i in range(n):
                row = f"{x[i]:>20.4f}\t{y[i]:>20.4f}\t{x2[i]:>20.4f}\t{xy[i]:>20.4f} "
                if check:
                    row += f"\t{syn[i] if syn[i] is not None else 'N/A':20}\t{w[i] if w[i] is not None else 'N/A':>20}"
                file.write(row + "\n")

            print(f"Results saved to '{file_name}'")  # Confirmation of file saving
            
    def show(self, **kwargs):
        # Display the data and fitted curve using matplotlib
        x_axis = kwargs.get('x_axis', 'x')  # Label for x-axis
        y_axis = kwargs.get('y_axis', 'y')  # Label for y-axis
        title = kwargs.get('title', 'Least Squares Best Fit')  # Plot title
        log_x = kwargs.get('logx', False)  # Logarithmic scale for x-axis
        log_y = kwargs.get('logy', False)  # Logarithmic scale for y-axis
        log = kwargs.get('log', False)  # Logarithmic scale for both axes
        col = kwargs.get('color', 'b')  # Color for the fitted line
        
        n = self.m_data.shape[0]  # Number of data points
        data = self.m_data  # Access the data
        x, y = sp.symbols('x y')  # Define symbolic variables for the equation

        # Define the functional form based on the log settings
        if log:
            fx = sp.exp(sp.log(x) * self.a + self.b)
        elif log_x:
            fx = sp.log(x) * self.a + self.b
        elif log_y:
            fx = sp.exp(x * self.a + self.b)
        else:
            fx = self.a * x + self.b

        f = sp.lambdify(x, fx, 'numpy')  # Create a numerical function for evaluation

        # Plotting the original data points with error bars
        for i in range(n):
            if i == 0:
                plt.errorbar(data[i][0], data[i][2], xerr=data[i][1], yerr=data[i][3], fmt='o', 
                             label="Data", ecolor='red', color='k', capsize=2)
            else:
                plt.errorbar(data[i][0], data[i][2], xerr=data[i][1], yerr=data[i][3], fmt='o', 
                             ecolor='red', color='k', capsize=2)
                             
        # Prepare x values for the fitted line
        xmin = np.min(data[:, 0])  # Minimum x value
        xmax = np.max(data[:, 0])  # Maximum x value

        xs = np.linspace(xmin * 0.8, xmax * 1.2, 100)  # Generate x values for the fit line
        plt.plot(xs, f(xs), color=col, label='Best Fit')  # Plot the fitted line

        # Labeling the axes and title
        plt.xlabel(f'${x_axis}$')
        plt.ylabel(f'${y_axis}$')
        plt.title(f'${title}$')

        # Add text annotation to the plot for coefficients
        plt.text(0.05, 0.95, f"$a = {self.a:.4f} \pm {self.sa:.4f}$\n"
                            f"$b = {self.b:.4f} \pm {self.sb:.4f}$\n"
                            f"Equation: $y = {sp.latex(fx)}$",
                transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        # Handle logarithmic axes if specified
        if log:
            plt.loglog()
        elif log_x:
            plt.semilogx()
        elif log_y:
            plt.semilogy()

        plt.legend(loc='best')  # Show legend
        plt.grid(True)  # Enable grid
        plt.show()  # Display the plot
        plt.savefig('plot_of_least_squares.pdf')  # Save the plot as PDF

    def transform(self):
        # Dummy transform function (currently does nothing)
        data = self.m_data
        for i in range(data.shape[0]):
            data[i][0] = data[i][0]  # No operation
            data[i][1] = data[i][1]  # No operation
            data[i][2] = data[i][2]  # No operation
            data[i][3] = data[i][3]  # No operation

    def revert(self):
        # Dummy revert function (currently does nothing)
        data = self.m_data
        for i in range(data.shape[0]):
            data[i][0] = data[i][0]  # No operation
            data[i][1] = data[i][1]  # No operation
            data[i][2] = data[i][2]  # No operation
            data[i][3] = data[i][3]  # No operation
        

def main():
    # Example usage of the Lab class
    a = Lab(file='test.txt')  # Create an instance of Lab, reading data from 'test.txt'
    print(a)  # Print the instance data
    a.least_square()  # Perform least squares fitting
    # a.show()  # (Commented out) Call to show the plot
    a.least_square(True, a.a)  # Perform least squares fitting with uncertainty check
    a.show()  # Show the fitted data and curve

if __name__ == '__main__':
    main()  # Execute the main function
