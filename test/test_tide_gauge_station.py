from datetime import datetime
from unittest import TestCase

import numpy as np

from src.tide_gauge_station import TideGaugeStation


class TestTideGaugeStation(TestCase):
    def test_correct_reference_datum(self):
        tide_gauge_station = TideGaugeStation(1, "test", 0, 0, {}, {})
        date1 = datetime(2020, 1, 15)
        date2 = datetime(2020, 2, 15)
        date3 = datetime(2020, 3, 15)
        date4 = datetime(2020, 4, 15)
        tide_gauge_station.timeseries[date1] = 1
        tide_gauge_station.timeseries[date2] = 2
        tide_gauge_station.timeseries[date3] = 3
        tide_gauge_station.timeseries[date4] = 4
        np_date1 = np.datetime64(date1)
        np_date2 = np.datetime64(date2)
        np_date3 = np.datetime64(date3)
        np_date4 = np.datetime64(date4)
        reference_timeseries = [np_date1, np_date2, np_date3, np_date4]
        reference_timeseries_value = [-3, -2, -1, 0]
        tide_gauge_station.correct_reference_datum(reference_timeseries_value, reference_timeseries)

        self.assertEqual(tide_gauge_station.timeseries[date1], 1)
        self.assertEqual(tide_gauge_station.timeseries[date2], 2)
        self.assertEqual(tide_gauge_station.timeseries[date3], 3)
        self.assertEqual(tide_gauge_station.timeseries[date4], 4)
        self.assertEqual(tide_gauge_station.timeseries_corrected_reference_datum[date1], -3)
        self.assertEqual(tide_gauge_station.timeseries_corrected_reference_datum[date2], -2)
        self.assertEqual(tide_gauge_station.timeseries_corrected_reference_datum[date3], -1)
        self.assertEqual(tide_gauge_station.timeseries_corrected_reference_datum[date4], 0)

    def test_correct_reference_datum_longer_time_series(self):
        tide_gauge_station = TideGaugeStation(1, "test", 0, 0, {}, {})
        date0 = datetime(2019, 1, 15)
        date1 = datetime(2020, 1, 15)
        date2 = datetime(2020, 2, 15)
        date3 = datetime(2020, 3, 15)
        date4 = datetime(2020, 4, 15)
        date5 = datetime(2020, 5, 15)
        tide_gauge_station.timeseries[date0] = 0
        tide_gauge_station.timeseries[date1] = 1
        tide_gauge_station.timeseries[date2] = 2
        tide_gauge_station.timeseries[date3] = 3
        tide_gauge_station.timeseries[date4] = 4
        tide_gauge_station.timeseries[date5] = 5
        np_date1 = np.datetime64(date1)
        np_date2 = np.datetime64(date2)
        np_date3 = np.datetime64(date3)
        np_date4 = np.datetime64(date4)
        reference_timeseries = [np_date1, np_date2, np_date3, np_date4]
        reference_timeseries_value = [-3, -2, -1, 0]
        tide_gauge_station.correct_reference_datum(reference_timeseries_value, reference_timeseries)

        self.assertEqual(tide_gauge_station.timeseries[date0], 0)
        self.assertEqual(tide_gauge_station.timeseries[date1], 1)
        self.assertEqual(tide_gauge_station.timeseries[date2], 2)
        self.assertEqual(tide_gauge_station.timeseries[date3], 3)
        self.assertEqual(tide_gauge_station.timeseries[date4], 4)
        self.assertEqual(tide_gauge_station.timeseries[date5], 5)
        self.assertEqual(tide_gauge_station.timeseries_corrected_reference_datum[date0], -4)
        self.assertEqual(tide_gauge_station.timeseries_corrected_reference_datum[date1], -3)
        self.assertEqual(tide_gauge_station.timeseries_corrected_reference_datum[date2], -2)
        self.assertEqual(tide_gauge_station.timeseries_corrected_reference_datum[date3], -1)
        self.assertEqual(tide_gauge_station.timeseries_corrected_reference_datum[date4], 0)
        self.assertEqual(tide_gauge_station.timeseries_corrected_reference_datum[date5], 1)

    def test_correct_reference_datum_with_real_values(self):
        tide_gauge_station = TideGaugeStation(1, "test", 0, 0, {}, {})
        dates = [datetime(1993, 1, 16), datetime(1993, 2, 15), datetime(1993, 3, 18),
                 datetime(1993, 4, 17), datetime(1993, 5, 17), datetime(1993, 6, 17),
                 datetime(1993, 7, 17), datetime(1993, 8, 17), datetime(1993, 9, 16),
                 datetime(1993, 10, 16), datetime(1993, 11, 16), datetime(1993, 12, 16),
                 datetime(1994, 1, 16), datetime(1994, 2, 15), datetime(1994, 3, 18),
                 datetime(1994, 4, 17), datetime(1994, 5, 17), datetime(1994, 6, 17),
                 datetime(1994, 7, 17), datetime(1994, 8, 17), datetime(1994, 9, 16),
                 datetime(1994, 10, 16), datetime(1994, 11, 16), datetime(1994, 12, 16),
                 datetime(1995, 1, 16), datetime(1995, 2, 15), datetime(1995, 3, 18),
                 datetime(1995, 4, 17), datetime(1995, 5, 17), datetime(1995, 6, 17),
                 datetime(1995, 7, 17), datetime(1995, 8, 17), datetime(1995, 9, 16),
                 datetime(1995, 10, 16), datetime(1995, 11, 16), datetime(1995, 12, 16),
                 datetime(1996, 1, 16), datetime(1996, 2, 15), datetime(1996, 3, 17),
                 datetime(1996, 4, 16), datetime(1996, 5, 17), datetime(1996, 6, 16),
                 datetime(1996, 7, 17), datetime(1996, 8, 16), datetime(1996, 9, 16),
                 datetime(1996, 10, 16), datetime(1996, 11, 16), datetime(1996, 12, 16),
                 datetime(1997, 1, 16), datetime(1997, 2, 15), datetime(1997, 3, 18),
                 datetime(1997, 4, 17), datetime(1997, 5, 17), datetime(1997, 6, 17),
                 datetime(1997, 7, 17), datetime(1997, 8, 17), datetime(1997, 9, 16),
                 datetime(1997, 10, 16), datetime(1997, 11, 16), datetime(1997, 12, 16),
                 datetime(1998, 1, 16), datetime(1998, 2, 15), datetime(1998, 3, 18),
                 datetime(1998, 4, 17), datetime(1998, 5, 17), datetime(1998, 6, 17),
                 datetime(1998, 7, 17), datetime(1998, 8, 17), datetime(1998, 9, 16),
                 datetime(1998, 10, 16), datetime(1998, 11, 16), datetime(1998, 12, 16),
                 datetime(1999, 1, 16), datetime(1999, 2, 15), datetime(1999, 3, 18),
                 datetime(1999, 4, 17), datetime(1999, 5, 17), datetime(1999, 6, 17),
                 datetime(1999, 7, 17), datetime(1999, 8, 17), datetime(1999, 9, 16),
                 datetime(1999, 10, 16), datetime(1999, 11, 16), datetime(1999, 12, 16),
                 datetime(2000, 1, 16), datetime(2000, 2, 15), datetime(2000, 3, 17),
                 datetime(2000, 4, 16), datetime(2000, 5, 17), datetime(2000, 6, 16),
                 datetime(2000, 7, 17), datetime(2000, 8, 16), datetime(2000, 9, 16),
                 datetime(2000, 10, 16), datetime(2000, 11, 16), datetime(2000, 12, 16),
                 datetime(2001, 1, 16), datetime(2001, 2, 15), datetime(2001, 3, 18),
                 datetime(2001, 4, 17), datetime(2001, 5, 17), datetime(2001, 6, 17),
                 datetime(2001, 7, 17), datetime(2001, 8, 17), datetime(2001, 9, 16),
                 datetime(2001, 10, 16), datetime(2001, 11, 16), datetime(2001, 12, 16),
                 datetime(2002, 1, 16), datetime(2002, 2, 15), datetime(2002, 3, 18),
                 datetime(2002, 4, 17), datetime(2002, 5, 17), datetime(2002, 6, 17),
                 datetime(2002, 7, 17), datetime(2002, 8, 17), datetime(2002, 9, 16),
                 datetime(2002, 10, 16), datetime(2002, 11, 16), datetime(2002, 12, 16),
                 datetime(2003, 1, 16), datetime(2003, 2, 15), datetime(2003, 3, 18),
                 datetime(2003, 4, 17), datetime(2003, 5, 17), datetime(2003, 6, 17),
                 datetime(2003, 7, 17), datetime(2003, 8, 17), datetime(2003, 9, 16),
                 datetime(2003, 10, 16), datetime(2003, 11, 16), datetime(2003, 12, 16),
                 datetime(2004, 1, 16), datetime(2004, 2, 15), datetime(2004, 3, 17),
                 datetime(2004, 4, 16), datetime(2004, 5, 17), datetime(2004, 6, 16),
                 datetime(2004, 7, 17), datetime(2004, 8, 16), datetime(2004, 9, 16),
                 datetime(2004, 10, 16), datetime(2004, 11, 16), datetime(2004, 12, 16),
                 datetime(2005, 1, 16), datetime(2005, 2, 15), datetime(2005, 3, 18),
                 datetime(2005, 4, 17), datetime(2005, 5, 17), datetime(2005, 6, 17),
                 datetime(2005, 7, 17), datetime(2005, 8, 17), datetime(2005, 9, 16),
                 datetime(2005, 10, 16), datetime(2005, 11, 16), datetime(2005, 12, 16),
                 datetime(2006, 1, 16), datetime(2006, 2, 15), datetime(2006, 3, 18),
                 datetime(2006, 4, 17), datetime(2006, 5, 17), datetime(2006, 6, 17),
                 datetime(2006, 7, 17), datetime(2006, 8, 17), datetime(2006, 9, 16),
                 datetime(2006, 10, 16), datetime(2006, 11, 16), datetime(2006, 12, 16),
                 datetime(2007, 1, 16), datetime(2007, 2, 15), datetime(2007, 3, 18),
                 datetime(2007, 4, 17), datetime(2007, 5, 17), datetime(2007, 6, 17),
                 datetime(2007, 7, 17), datetime(2007, 8, 17), datetime(2007, 9, 16),
                 datetime(2007, 10, 16), datetime(2007, 11, 16), datetime(2007, 12, 16),
                 datetime(2008, 1, 16), datetime(2008, 2, 15), datetime(2008, 3, 17),
                 datetime(2008, 4, 16), datetime(2008, 5, 17), datetime(2008, 6, 16),
                 datetime(2008, 7, 17), datetime(2008, 8, 16), datetime(2008, 9, 16),
                 datetime(2008, 10, 16), datetime(2008, 11, 16), datetime(2008, 12, 16),
                 datetime(2009, 1, 16), datetime(2009, 2, 15), datetime(2009, 3, 18),
                 datetime(2009, 4, 17), datetime(2009, 5, 17), datetime(2009, 6, 17),
                 datetime(2009, 7, 17), datetime(2009, 8, 17), datetime(2009, 9, 16),
                 datetime(2009, 10, 16), datetime(2009, 11, 16), datetime(2009, 12, 16),
                 datetime(2010, 1, 16), datetime(2010, 2, 15), datetime(2010, 3, 18),
                 datetime(2010, 4, 17), datetime(2010, 5, 17), datetime(2010, 6, 17),
                 datetime(2010, 7, 17), datetime(2010, 8, 17), datetime(2010, 9, 16),
                 datetime(2010, 10, 16), datetime(2010, 11, 16), datetime(2010, 12, 16),
                 datetime(2011, 1, 16), datetime(2011, 2, 15), datetime(2011, 3, 18),
                 datetime(2011, 4, 17), datetime(2011, 5, 17), datetime(2011, 6, 17),
                 datetime(2011, 7, 17), datetime(2011, 8, 17), datetime(2011, 9, 16),
                 datetime(2011, 10, 16), datetime(2011, 11, 16), datetime(2011, 12, 16),
                 datetime(2012, 1, 16), datetime(2012, 2, 15), datetime(2012, 3, 17),
                 datetime(2012, 4, 16), datetime(2012, 5, 17), datetime(2012, 6, 16),
                 datetime(2012, 7, 17), datetime(2012, 8, 16), datetime(2012, 9, 16),
                 datetime(2012, 10, 16), datetime(2012, 11, 16), datetime(2012, 12, 16),
                 datetime(2013, 1, 16), datetime(2013, 2, 15), datetime(2013, 3, 18),
                 datetime(2013, 4, 17), datetime(2013, 5, 17), datetime(2013, 6, 17),
                 datetime(2013, 7, 17), datetime(2013, 8, 17), datetime(2013, 9, 16),
                 datetime(2013, 10, 16), datetime(2013, 11, 16), datetime(2013, 12, 16),
                 datetime(2014, 1, 16), datetime(2014, 2, 15), datetime(2014, 3, 18),
                 datetime(2014, 4, 17), datetime(2014, 5, 17), datetime(2014, 6, 17),
                 datetime(2014, 7, 17), datetime(2014, 8, 17), datetime(2014, 9, 16),
                 datetime(2014, 10, 16), datetime(2014, 11, 16), datetime(2014, 12, 16),
                 datetime(2015, 1, 16), datetime(2015, 2, 15), datetime(2015, 3, 18),
                 datetime(2015, 4, 17), datetime(2015, 5, 17), datetime(2015, 6, 17),
                 datetime(2015, 7, 17), datetime(2015, 8, 17), datetime(2015, 9, 16),
                 datetime(2015, 10, 16), datetime(2015, 11, 16), datetime(2015, 12, 16),
                 datetime(2016, 1, 16), datetime(2016, 2, 15), datetime(2016, 3, 17),
                 datetime(2016, 4, 16), datetime(2016, 5, 17), datetime(2016, 6, 16),
                 datetime(2016, 7, 17), datetime(2016, 8, 16), datetime(2016, 9, 16),
                 datetime(2016, 10, 16), datetime(2016, 11, 16), datetime(2016, 12, 16),
                 datetime(2017, 1, 16), datetime(2017, 2, 15), datetime(2017, 3, 18),
                 datetime(2017, 4, 17), datetime(2017, 5, 17), datetime(2017, 6, 17),
                 datetime(2017, 7, 17), datetime(2017, 8, 17), datetime(2017, 9, 16),
                 datetime(2017, 10, 16), datetime(2017, 11, 16), datetime(2017, 12, 16),
                 datetime(2018, 1, 16), datetime(2018, 2, 15), datetime(2018, 3, 18),
                 datetime(2018, 4, 17), datetime(2018, 5, 17), datetime(2018, 6, 17),
                 datetime(2018, 7, 17), datetime(2018, 8, 17), datetime(2018, 9, 16),
                 datetime(2018, 10, 16), datetime(2018, 11, 16), datetime(2018, 12, 16),
                 datetime(2019, 1, 16), datetime(2019, 2, 15), datetime(2019, 3, 18),
                 datetime(2019, 4, 17), datetime(2019, 5, 17), datetime(2019, 6, 17),
                 datetime(2019, 7, 17), datetime(2019, 8, 17), datetime(2019, 9, 16),
                 datetime(2019, 10, 16), datetime(2019, 11, 16), datetime(2019, 12, 16),
                 datetime(2020, 1, 16), datetime(2020, 2, 15), datetime(2020, 3, 17),
                 datetime(2020, 4, 16), datetime(2020, 5, 17), datetime(2020, 6, 16),
                 datetime(2020, 7, 17), datetime(2020, 8, 16), datetime(2020, 9, 16),
                 datetime(2020, 10, 16), datetime(2020, 11, 16), datetime(2020, 12, 16),
                 datetime(2021, 1, 16), datetime(2021, 2, 15), datetime(2021, 3, 18),
                 datetime(2021, 4, 17), datetime(2021, 5, 17), datetime(2021, 6, 17),
                 datetime(2021, 7, 17), datetime(2021, 8, 17), datetime(2021, 9, 16),
                 datetime(2021, 10, 16), datetime(2021, 11, 16), datetime(2021, 12, 16),
                 datetime(2022, 1, 16), datetime(2022, 2, 15), datetime(2022, 3, 18),
                 datetime(2022, 4, 17), datetime(2022, 5, 17), datetime(2022, 6, 17),
                 datetime(2022, 7, 17), datetime(2022, 8, 17), datetime(2022, 9, 16),
                 datetime(2022, 10, 16), datetime(2022, 11, 16), datetime(2022, 12, 16),
                 datetime(2023, 1, 16), datetime(2023, 2, 15), datetime(2023, 3, 18),
                 datetime(2023, 4, 17), datetime(2023, 5, 17)]

        values = [7.138, 6.953, 6.983, 7.11, 7.106, 7.049, 7.012, 7.0, 7.19, 7.198, 7.123, 7.135, 7.091, 7.171, 7.012,
                  7.004, 7.149, 7.02, 7.041, 7.103, 7.157, 7.191, 7.194, 7.201, 7.164, 7.192, 7.055, 7.0, 7.05, 6.989,
                  7.08, 7.025, 7.128, 7.171, 7.245, 7.226, 7.37, 7.085, 7.113, 7.059, 7.114, 6.987, 7.02, 7.044, 7.057,
                  7.109, 7.144, 7.143, 7.106, 7.069, 6.969, 6.977, 7.136, 7.195, 7.004, 7.098, 7.071, 7.174, 7.347,
                  7.214, 7.182, 7.036, 6.999, 7.222, 7.055, 7.1, 7.072, 7.013, 7.168, 7.144, 7.108, 7.078, 7.148, 6.961,
                  7.073, 7.076, 7.082, 7.05, 7.048, 7.148, 7.221, 7.212, 7.096, 7.158, 6.988, 7.048, 6.976, 7.215,
                  7.109, 7.024, 7.076, 7.047, 7.144, 7.161, 7.357, 7.426, 7.355, 7.116, 7.31, 7.074, 7.039, 7.017,
                  7.077, 7.084, 7.087, 7.254, 7.027, 7.045, 7.147, 7.148, 7.075, 7.074, 7.118, 7.039, 7.029, 7.044,
                  7.101, 7.219, 7.342, 7.282, 7.145, 7.094, 7.122, 7.148, 7.09, 7.108, 7.085, 7.075, 7.078, 7.174,
                  7.217, 7.144, 7.173, 7.07, 7.068, 7.114, 7.092, 7.065, 7.065, 7.189, 7.095, 7.327, 7.047, 7.066,
                  6.992, 6.943, 7.048, 7.085, 7.054, 7.065, 7.071, 7.04, 7.089, 7.177, 7.155, 7.084, 7.008, 7.04, 7.115,
                  7.027, 7.072, 7.031, 7.053, 7.061, 7.173, 7.31, 7.258, 7.135, 7.106, 7.23, 7.061, 7.018, 7.111, 7.176,
                  7.142, 7.091, 7.051, 7.105, 7.067, 7.089, 7.175, 7.11, 7.124, 7.146, 7.168, 7.083, 7.127, 7.132,
                  7.137, 7.141, 7.138, 7.086, 7.182, 7.117, 7.057, 7.103, 7.044, 7.132, 7.135, 7.092, 7.043, 7.164,
                  7.321, 7.3, 7.161, 7.273, 7.137, 7.077, 7.072, 7.093, 7.085, 7.095, 7.135, 7.228, 7.291, 7.186, 7.123,
                  7.15, 7.044, 7.072, 7.064, 7.093, 7.088, 7.108, 7.127, 7.141, 7.218, 7.128, 7.026, 6.875, 6.942, 7.19,
                  7.114, 7.156, 7.124, 7.178, 7.129, 7.264, 7.256, 7.196, 7.187, 7.061, 7.238, 7.142, 7.1, 7.061, 7.094,
                  7.105, 7.113, 7.271, -99999.0, 7.198, 7.339, 7.36, 7.113, 7.17, 7.118, 7.11, 7.112, 7.137, 7.165,
                  7.253, 7.4, 7.103, 7.117, 7.05, 6.999, 7.06, 7.114, 7.049, 7.12, 7.156, 7.152, 7.215, 7.187, 7.249,
                  7.309, 7.197, 7.087, 7.166, 7.167, 7.134, 7.093, 7.095, 7.129, 7.154, 7.203, 7.098, 7.062, 7.165,
                  7.109, 7.033, 7.168, 7.149, 7.142, 7.125, 7.163, 7.145, 7.136, 7.145, 7.198, 7.071, 7.339, 7.208,
                  7.114, 7.124, 7.126, 7.106, 7.1, 7.141, 7.305, 7.22, 7.079, 7.119, 7.1, 7.202, 7.115, 7.157, 7.158,
                  7.153, 7.138, 7.287, 7.384, 7.326, 7.211, 7.232, 7.16, 7.198, 7.123, 7.23, 7.117, 7.209, 7.15,
                  -99999.0, -99999.0, 7.359, 7.223, 7.305, 7.099, 7.113, 7.203, 7.135, 7.17, 7.14, 7.19, 7.234, 7.153,
                  7.23, 7.082, 7.114, 7.12, -99999.0, -99999.0, 7.175, 7.088, 7.125, 7.21, 7.261, 7.391, 7.33, 7.237,
                  7.033, 7.277, 7.183, 7.098]

        np_dates = [np.datetime64(date) for date in dates]
        reference_timeseries_values = [np.float64(0.0526), np.float64(-0.0022), np.float64(-0.0181),
                                       np.float64(-0.0357), np.float64(-0.053700000000000005), np.float64(-0.0152),
                                       np.float64(-0.06670000000000001), np.float64(-0.030500000000000003),
                                       np.float64(0.029400000000000003), np.float64(0.0742),
                                       np.float64(0.06570000000000001), np.float64(0.013900000000000001),
                                       np.float64(-0.0373), np.float64(0.0241), np.float64(-0.019), np.float64(-0.0763),
                                       np.float64(-0.0235), np.float64(-0.0143), np.float64(-0.06330000000000001),
                                       np.float64(-0.045000000000000005), np.float64(0.029300000000000003),
                                       np.float64(0.047), np.float64(0.1068), np.float64(0.1015), np.float64(0.055),
                                       np.float64(0.0317), np.float64(-0.055200000000000006),
                                       np.float64(-0.058100000000000006), np.float64(-0.0179), np.float64(-0.0362),
                                       np.float64(-0.0322), np.float64(-0.0223), np.float64(-0.0007),
                                       np.float64(0.0969), np.float64(0.1495), np.float64(0.0955), np.float64(0.1507),
                                       np.float64(-0.0279), np.float64(0.0258), np.float64(-0.0198),
                                       np.float64(-0.023200000000000002), np.float64(-0.0366), np.float64(-0.0373),
                                       np.float64(-0.0217), np.float64(0.0063), np.float64(0.051500000000000004),
                                       np.float64(0.053200000000000004), np.float64(0.049100000000000005),
                                       np.float64(0.06420000000000001), np.float64(0.047900000000000005),
                                       np.float64(0.024300000000000002), np.float64(-0.041), np.float64(0.0108),
                                       np.float64(0.0089), np.float64(-0.042800000000000005),
                                       np.float64(0.020800000000000003), np.float64(0.0644),
                                       np.float64(0.10210000000000001), np.float64(0.1671), np.float64(0.0952),
                                       np.float64(0.07740000000000001), np.float64(0.0618),
                                       np.float64(0.021400000000000002), np.float64(0.0027),
                                       np.float64(-0.06620000000000001), np.float64(-0.020200000000000003),
                                       np.float64(-0.0178), np.float64(-0.0112), np.float64(0.0427), np.float64(0.0857),
                                       np.float64(0.068), np.float64(0.0292), np.float64(0.0521), np.float64(-0.0694),
                                       np.float64(-0.0304), np.float64(-0.060700000000000004),
                                       np.float64(-0.020900000000000002), np.float64(-0.041600000000000005),
                                       np.float64(-0.0195), np.float64(0.0173), np.float64(0.0627),
                                       np.float64(0.09630000000000001), np.float64(0.0637),
                                       np.float64(0.045700000000000005), np.float64(0.023700000000000002),
                                       np.float64(0.0466), np.float64(-0.0256), np.float64(0.0171), np.float64(-0.0022),
                                       np.float64(0.005200000000000001), np.float64(-0.045200000000000004),
                                       np.float64(0.005), np.float64(0.0011), np.float64(0.0644),
                                       np.float64(0.08600000000000001), np.float64(0.2127),
                                       np.float64(0.13920000000000002), np.float64(0.0415), np.float64(0.0675),
                                       np.float64(-0.0218), np.float64(-0.0024000000000000002),
                                       np.float64(-0.022000000000000002), np.float64(-0.0103), np.float64(0.0262),
                                       np.float64(0.0151), np.float64(0.1165), np.float64(0.0514), np.float64(0.0645),
                                       np.float64(0.10590000000000001), np.float64(0.0366), np.float64(0.0229),
                                       np.float64(0.0073), np.float64(-0.0236), np.float64(-0.0485), np.float64(-0.029),
                                       np.float64(0.040100000000000004), np.float64(0.0285),
                                       np.float64(0.07780000000000001), np.float64(0.15760000000000002),
                                       np.float64(0.1337), np.float64(0.07440000000000001), np.float64(0.0218),
                                       np.float64(0.0712), np.float64(0.0655), np.float64(0.0219), np.float64(0.0308),
                                       np.float64(0.036500000000000005), np.float64(0.051500000000000004),
                                       np.float64(0.0809), np.float64(0.08650000000000001),
                                       np.float64(0.13340000000000002), np.float64(0.08560000000000001),
                                       np.float64(0.061000000000000006), np.float64(0.0298), np.float64(0.0361),
                                       np.float64(-0.0222), np.float64(-0.0131), np.float64(-0.0159),
                                       np.float64(-0.0375), np.float64(0.0393), np.float64(0.08270000000000001),
                                       np.float64(0.0975), np.float64(0.0689), np.float64(0.0651), np.float64(-0.0025),
                                       np.float64(-0.0392), np.float64(-0.028900000000000002), np.float64(-0.0177),
                                       np.float64(-0.0257), np.float64(-0.0125), np.float64(-0.0074),
                                       np.float64(0.0004), np.float64(0.0275), np.float64(0.0829),
                                       np.float64(0.11950000000000001), np.float64(0.038700000000000005),
                                       np.float64(0.060700000000000004), np.float64(-0.0206), np.float64(-0.0037),
                                       np.float64(-0.0149), np.float64(-0.0181), np.float64(-0.0324),
                                       np.float64(-0.020300000000000002), np.float64(-0.0068000000000000005),
                                       np.float64(0.08170000000000001), np.float64(0.1527),
                                       np.float64(0.15960000000000002), np.float64(0.0971), np.float64(0.0579),
                                       np.float64(0.042300000000000004), np.float64(-0.0236), np.float64(-0.0257),
                                       np.float64(-0.023700000000000002), np.float64(0.023), np.float64(-0.0022),
                                       np.float64(-0.0054), np.float64(0.0099), np.float64(0.0956), np.float64(0.0342),
                                       np.float64(0.047400000000000005), np.float64(0.0835), np.float64(0.0489),
                                       np.float64(-0.0167), np.float64(-0.0032), np.float64(0.0067), np.float64(0.0028),
                                       np.float64(-0.013800000000000002), np.float64(0.0352), np.float64(0.0438),
                                       np.float64(0.08510000000000001), np.float64(0.0482),
                                       np.float64(0.053700000000000005), np.float64(0.0853), np.float64(0.0334),
                                       np.float64(0.01), np.float64(-0.027100000000000003), np.float64(-0.0273),
                                       np.float64(0.0577), np.float64(0.0132), np.float64(0.0553), np.float64(0.0454),
                                       np.float64(0.0791), np.float64(0.1188), np.float64(0.1179), np.float64(0.0614),
                                       np.float64(0.08070000000000001), np.float64(0.026600000000000002),
                                       np.float64(0.014400000000000001), np.float64(-0.027200000000000002),
                                       np.float64(0.0115), np.float64(0.0018000000000000002), np.float64(0.0002),
                                       np.float64(0.058300000000000005), np.float64(0.1169), np.float64(0.1013),
                                       np.float64(0.0882), np.float64(0.0776), np.float64(0.0585), np.float64(0.0085),
                                       np.float64(0.025), np.float64(-0.0018000000000000002), np.float64(0.0356),
                                       np.float64(0.0126), np.float64(0.0391), np.float64(0.0431), np.float64(0.1005),
                                       np.float64(0.1336), np.float64(0.0947), np.float64(0.055400000000000005),
                                       np.float64(-0.0077), np.float64(-0.0261), np.float64(0.014100000000000001),
                                       np.float64(0.016300000000000002), np.float64(0.0261),
                                       np.float64(0.031200000000000002), np.float64(0.06620000000000001),
                                       np.float64(0.07200000000000001), np.float64(0.0941),
                                       np.float64(0.12290000000000001), np.float64(0.09580000000000001),
                                       np.float64(0.0779), np.float64(0.007), np.float64(0.025900000000000003),
                                       np.float64(0.0252), np.float64(-0.0177), np.float64(-0.0198),
                                       np.float64(0.0078000000000000005), np.float64(0.04), np.float64(0.0425),
                                       np.float64(0.11860000000000001), np.float64(0.062200000000000005),
                                       np.float64(0.09040000000000001), np.float64(0.098), np.float64(0.0651),
                                       np.float64(0.038200000000000005), np.float64(0.0436),
                                       np.float64(-0.011000000000000001), np.float64(0.0074), np.float64(0.0114),
                                       np.float64(0.015700000000000002), np.float64(0.1018), np.float64(0.1582),
                                       np.float64(0.1884), np.float64(0.06960000000000001), np.float64(0.0403),
                                       np.float64(-0.019700000000000002), np.float64(-0.0216), np.float64(0.0177),
                                       np.float64(0.0256), np.float64(0.013900000000000001), np.float64(0.0388),
                                       np.float64(0.0645), np.float64(0.0684), np.float64(0.1423), np.float64(0.1497),
                                       np.float64(0.1553), np.float64(0.1453), np.float64(0.050800000000000005),
                                       np.float64(-0.011600000000000001), np.float64(0.052500000000000005),
                                       np.float64(0.031100000000000003), np.float64(0.031), np.float64(0.0172),
                                       np.float64(0.0727), np.float64(0.0926), np.float64(0.11170000000000001),
                                       np.float64(0.1341), np.float64(0.1253), np.float64(0.09090000000000001),
                                       np.float64(0.054), np.float64(0.0253), np.float64(0.0097), np.float64(0.0531),
                                       np.float64(0.0388), np.float64(0.055600000000000004), np.float64(0.0654),
                                       np.float64(0.0902), np.float64(0.1351), np.float64(0.0813), np.float64(0.0425),
                                       np.float64(0.0942), np.float64(0.025), np.float64(0.10690000000000001),
                                       np.float64(0.059000000000000004), np.float64(0.0315),
                                       np.float64(0.027100000000000003), np.float64(0.0183), np.float64(0.0599),
                                       np.float64(0.0805), np.float64(0.1163), np.float64(0.1443),
                                       np.float64(0.15990000000000001), np.float64(0.051300000000000005),
                                       np.float64(0.08950000000000001), np.float64(0.0601),
                                       np.float64(0.044000000000000004), np.float64(0.0512), np.float64(0.0359),
                                       np.float64(0.0631), np.float64(0.09050000000000001),
                                       np.float64(0.10310000000000001), np.float64(0.14600000000000002),
                                       np.float64(0.1394), np.float64(0.1269), np.float64(0.151), np.float64(0.1449),
                                       np.float64(0.0697), np.float64(0.0809), np.float64(0.0651), np.float64(0.0738),
                                       np.float64(0.0431), np.float64(0.10640000000000001), np.float64(0.0847),
                                       np.float64(0.12710000000000002), np.float64(0.189),
                                       np.float64(0.15860000000000002), np.float64(0.09580000000000001),
                                       np.float64(0.153), np.float64(0.065), np.float64(0.0718), np.float64(0.0431),
                                       np.float64(0.0627), np.float64(0.0562), np.float64(0.0529), np.float64(0.1012),
                                       np.float64(0.1444), np.float64(0.116), np.float64(0.1121), np.float64(0.1288),
                                       np.float64(0.0772), np.float64(0.0455), np.float64(0.053200000000000004),
                                       np.float64(0.040400000000000005), np.float64(0.053500000000000006),
                                       np.float64(0.0308), np.float64(0.060200000000000004), np.float64(0.0767),
                                       np.float64(0.17200000000000001), np.float64(0.24100000000000002),
                                       np.float64(0.20720000000000002), np.float64(0.14300000000000002),
                                       np.float64(0.09860000000000001), np.float64(0.09570000000000001),
                                       np.float64(0.07980000000000001), np.float64(0.0431)]
        tide_gauge_station.timeseries = dict(zip(dates, values))
        tide_gauge_station.correct_reference_datum(reference_timeseries_values, np_dates)

        result = np.array(list(tide_gauge_station.timeseries_corrected_reference_datum.values()))
        valid_mask = result != -99999
        reference_array = np.array(reference_timeseries_values)
        assert np.isclose(result[valid_mask], reference_array[valid_mask], atol=0.5).all()
