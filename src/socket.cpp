/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <network/socket.hpp>

#include <stdio.h>

#include <iostream>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <cmath>

#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/select.h>

Socket::Socket() :
  m_sock ( -1 )
{

  memset ( &m_addr, 0, sizeof ( m_addr ) );

}

Socket::~Socket()
{
  if ( is_valid() )
    ::close ( m_sock );
}

bool Socket::create()
{
  m_sock = socket ( AF_INET, SOCK_STREAM, 0 );

  if ( ! is_valid() )
    return false;


  // TIME_WAIT - argh
  int on = 1;
  if ( setsockopt ( m_sock, SOL_SOCKET, SO_REUSEADDR, ( const char* ) &on, sizeof ( on ) ) == -1 )
    return false;

  return true;
}


bool Socket::bind ( const char * address, const int port )
{

  if ( ! is_valid() )
  {
    return false;
  }

  m_addr.sin_family = AF_INET;
  m_addr.sin_port = htons ( port );
  m_addr.sin_addr.s_addr = htonl(INADDR_ANY);

  if (strcmp(address, "any") == 0)
    m_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  else
  {
    m_addr.sin_addr.s_addr = inet_addr (address);
    // if we didn't parse the address as an IP address
    if (m_addr.sin_addr.s_addr == -1)
    {
      struct hostent * hp = gethostbyname (address);
      memcpy(&(m_addr.sin_addr.s_addr), hp->h_addr, hp->h_length);
    }
  }

  int bind_return = ::bind ( m_sock,
           ( struct sockaddr * ) &m_addr,
           sizeof ( m_addr ) );


  if ( bind_return == -1 )
    return false;

  return true;
}


bool Socket::listen() const
{
  if ( ! is_valid() )
    return false;

  int listen_return = ::listen ( m_sock, MAXCONNECTIONS );

  if ( listen_return == -1 )
    return false;

  return true;
}


bool Socket::accept ( Socket& new_Socket ) const
{
  int addr_length = sizeof ( m_addr );
  new_Socket.m_sock = ::accept ( m_sock, ( sockaddr * ) &m_addr, ( socklen_t * ) &addr_length );

  if ( new_Socket.m_sock <= 0 )
    return false;
  else
    return true;
}

bool Socket::send ( const std::string s ) const
{
  int status = ::send ( m_sock, s.c_str(), s.size(), MSG_NOSIGNAL );
  if ( status == -1 )
    return false;
  else
    return true;
}

int Socket::recv ( std::string& s ) const
{
  char buf [ MAXRECV + 1 ];

  s = "";

  memset ( buf, 0, MAXRECV + 1 );

  int status = ::recv ( m_sock, buf, MAXRECV, 0 );

  if ( status == -1 )
  {
    std::cout << "status == -1   errno == " << errno << "  in Socket::recv\n";
    return 0;
  }
  else if ( status == 0 )
  {
    return 0;
  }
  else
  {
    s = buf;
    return status;
  }
}

bool Socket::connect ( const std::string address, const int port )
{
  if ( ! is_valid() ) return false;

  m_addr.sin_family = AF_INET;
  m_addr.sin_port = htons ( port );
  m_addr.sin_addr.s_addr = inet_addr (address.c_str());

  // if we didn't parse the address as an IP address
  if (m_addr.sin_addr.s_addr == -1)
  {
    struct hostent * hp = gethostbyname (address.c_str());
    memcpy(&(m_addr.sin_addr.s_addr), hp->h_addr, hp->h_length);
  }

  if ( errno == EAFNOSUPPORT ) return false;

  int status = ::connect ( m_sock, ( sockaddr * ) &m_addr, sizeof ( m_addr ) );

  if ( status == 0 )
    return true;
  else
    return false;
}

void Socket::set_non_blocking ( const bool b )
{
  int opts;

  opts = fcntl ( m_sock,
     F_GETFL );

  if ( opts < 0 )
    return;

  if ( b )
    opts = ( opts | O_NONBLOCK );
  else
    opts = ( opts & ~O_NONBLOCK );

  fcntl ( m_sock,
    F_SETFL,opts );

}

int Socket::select_timeout ( float sleep_secs )
{
  struct timeval timeout;
  fd_set *rdsp = NULL;
  fd_set readset;

  float whole_seconds = floor (sleep_secs);
  float micro_seconds = sleep_secs - whole_seconds;
  micro_seconds *= 100000;

  timeout.tv_sec = (long int) whole_seconds;
  timeout.tv_usec = (long int) micro_seconds;

  FD_ZERO (&readset);
  FD_SET (m_sock, &readset);
  rdsp = &readset;

  // select returns the number of file descriptors changed
  // 0 if timeout, -1 if error
  return select (m_sock+1, rdsp, NULL, NULL, &timeout);
}
