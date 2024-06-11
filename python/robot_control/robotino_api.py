import json

import requests


class Robotino4:
    def __init__(self, ip_address='192.168.0.1'):
        self.base_url = f'http://{ip_address}/'
        self.is_online = self.ping_robot()

    def _compose_url(self, name, prefix='data/'):
        return self.base_url + prefix + name

    def ping_robot(self):
        try:
            response = requests.get(self._compose_url('powermanagement'))
            return response.status_code == requests.codes.ok
        except requests.exceptions.RequestException:
            return False

    def get_odometry(self):
        try:
            response = requests.get(self._compose_url('odometry'))
            assert response.status_code == requests.codes.ok, 'Bad response'
            return response.json()
        except requests.exceptions.RequestException as e:
            print("Error:", e)

    def get_bumper_status(self):
        try:
            response = requests.get(self._compose_url('bumper'))
            assert response.status_code == requests.codes.ok, 'Bad response'
            data = response.json()
            return data['value']
        except requests.exceptions.RequestException as e:
            print("Error:", e)

    def get_distances(self):
        try:
            response = requests.get(self._compose_url('distancesensorarray'))
            assert response.status_code == requests.codes.ok, 'Bad response'
            return response.json()
        except requests.exceptions.RequestException as e:
            print("Error:", e)

    def get_camera_image(self):
        try:
            response = requests.get(self._compose_url('cam0', prefix=''))
            return response.content
        except requests.exceptions.RequestException as e:
            print("Error:", e)

    def get_currents(self):
        try:
            response = requests.get(self._compose_url('powermanagement'))
            assert response.status_code == requests.codes.ok, 'Bad response'
            return response.json()
        except requests.exceptions.RequestException as e:
            print("Error:", e)

    def set_omnidrive(self, vx=0, vy=0, omega=0):
        data = [vx, vy, omega]
        try:
            response = requests.post(url=self._compose_url('omnidrive'), data=json.dumps(data))
            return response.status_code == requests.codes.ok
        except requests.exceptions.RequestException as e:
            print("Error:", e)

    def stop_movement(self):
        self.set_omnidrive(0, 0, 0)