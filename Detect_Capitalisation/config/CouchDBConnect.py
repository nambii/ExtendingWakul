#!/usr/bin/python3
"""CouchDBConnect
Connect to couchdb
"""
import sys
import os
import json
import yaml

import couchdb

import config.core as core


class CouchDBConnect() :

    def connect_CouchDB():
        """Connect to CouchDB"""


        args = core.config('couchdb')
        db_str="tweets"
        # Construct a URL from the arguments
        protocol = 'http'
        # Create a safe URL that we can print (omits login/password)
        url = url_safe = '{protocol}://{ip}:{port}/'.format(
            protocol = protocol,
            ip = args['ip_address'],
            port = str(args['port'])
        )
        # Construct an updated URL if a login and password is supplied
        if ('login' in args) and ('password' in args):
            url = '{protocol}://{login}:{password}@{ip}:{port}/'.format(
                protocol = protocol,
                login = args['login'],
                password = args['password'],
                ip = args['ip_address'],
                port = str(args['port'])
            )
        # Attempt to connect to CouchDB server
        try:
            # Calling couchdb.Server will not throw an exception
            couch = couchdb.Server(url)
            # Attempt to GET from the CouchDB server to test connection
            couch.version()
            print("Connected to CouchDB server at " + url_safe)
        except ConnectionRefusedError:
            print("No CouchDB server at " + url_safe)
            raise
        except couchdb.http.Unauthorized as e:
            print("Connection to CouchDB server refused: " + str(e))
            raise
        except Exception as e:
            print(
                "Failed to connect to CouchDB server at "
                + url_safe
                + ". An unexpected exception was raised: "
                + str(e)
            )
            raise
        # Attempt to connect to CouchDB database
        try:
            _db = couch[db_str]

        except couchdb.http.ResourceNotFound:
            try:
                _db = couch.create(db_str)
            except couchdb.http.Unauthorized:
                raise
            except Exception as e:
                raise
        except couchdb.http.Unauthorized:
            raise
        except Exception as e:
            raise
        return couch
