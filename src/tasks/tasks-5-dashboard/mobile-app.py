import sys
import io
import folium
import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg

from PyQt5.QtWebEngineWidgets import QWebEngineView

class MainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()

        ### Add a title
        self.setWindowTitle("California Earthquake Safest Path")
        self.window_width, self.window_height = 800, 550
        self.setMinimumSize(self.window_width, self.window_height)

        self.setGeometry(100, 100, 600, 400)
        
        ## Set Layout
        self.setLayout(qtw.QVBoxLayout())

        ## Create A Label
        my_label = qtw.QLabel("Enter your Current Location:")
       

        ## Change font size of label
        my_label.setFont(qtg.QFont('Helvetica', 18))
        self.layout().addWidget(my_label)

        ## Change checkbox

        my_entry = qtw.QLineEdit()
        my_entry.setObjectName("name_field")
        my_entry.setText("")

        self.layout().addWidget(my_entry)

        # Create a button
        my_button = qtw.QPushButton("Search", clicked = lambda: press_it())
        # my_button.resize(100, 100)
        self.layout().addWidget(my_button)


        coordinate = (37.8199286, -122.4782551)
        m = folium.Map(
        	tiles='Stamen Terrain',
        	zoom_start=13,
        	location=coordinate
        )

        data = io.BytesIO()
        m.save(data, close_file=False)

        webView = QWebEngineView()
        webView.setHtml(data.getvalue().decode())
        self.layout().addWidget(webView)

        self.show()

        def press_it():
            # Add name to label
            my_label.setText(f'Hello {my_entry.text()}' )
            # Clear the entry box
            my_entry.setText("")

            coordinate = (34.2203803, -118.5642751)
            m = folium.Map(
                tiles='cartodbpositron',
                zoom_start=13,
                location=coordinate
            )

            folium.Marker([34.2203803, -118.5642751], 
              icon=folium.Icon(color="orange", icon="graduation-cap", prefix='fa')
             ).add_to(m)


            data = io.BytesIO()
            m.save(data, close_file=False)

            webView.setHtml(data.getvalue().decode())


app = qtw.QApplication([])
mw = MainWindow()

app.exec_()