/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022  Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#define RAPFI_MAJOR_VER    0
#define RAPFI_MINOR_VER    38
#define RAPFI_REVISION_VER 06

#define MACRO_STR(s)         #s
#define VERSION_STR(a, b, c) MACRO_STR(a) "." MACRO_STR(b) "." MACRO_STR(c)
#define CURRENT_VER          VERSION_STR(RAPFI_MAJOR_VER, RAPFI_MINOR_VER, RAPFI_REVISION_VER)

constexpr char EngineInfo[] = "name=\"Rapfi\", "
                              "version=\"" CURRENT_VER "\", "
                              "author=\"dblue(https://github.com/dhbloo), sigmoid(https://github.com/hzyhhzy)\", "
                              "country=\"China\"";
