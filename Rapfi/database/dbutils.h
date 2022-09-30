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

#include "dbclient.h"
#include "dbstorage.h"
#include "dbtypes.h"

#include <ostream>
#include <istream>

class Board;  // forward declaration

namespace Database {

/// Serialize all records in the database to the output stream.
void databaseToCSVFile(DBStorage &dbStorage, std::ostream &csvStream);

/// Merge two dbStorage from dbSrc to dbDst with the overwrite rule.
/// @return The number of records (over)written.
size_t mergeDatabase(DBStorage &dbDst, DBStorage &dbSrc, OverwriteRule owRule);

/// Split a database branch from dbSrc to dbDst.
/// @return The number of records spilted to dbDst.
size_t splitDatabase(DBStorage &dbSrc, DBStorage &dbDst, const Board &board, Rule rule);

/// Import a lib file into the database.
/// @return The number of records (over)written.
size_t
importLibToDatabase(DBStorage &dbDst, std::istream& libStream, Rule rule, int boardSize = 15);

}  // namespace Database
