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

#include "config.h"

const std::string Config::InternalConfig = R"internalConfig(
[requirement]
min_version = [0,34,2]

[model]
[model.eval]
model_type = 2
table2 = [
0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	36,	41,
	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	36,	41,
		0,	0,	0,	1,	1,	1,	2,	9,	9,	16,	36,	41,
			0,	0,	1,	1,	1,	2,	10,	10,	17,	38,	43,
				1,	2,	2,	2,	3,	11,	11,	19,	39,	44,
					3,	3,	3,	4,	13,	13,	20,	42,	47,
						3,	3,	4,	13,	13,	20,	42,	47,
							3,	4,	13,	13,	20,	42,	47,
								5,	14,	14,	22,	43,	49,
									25,	25,	34,	57,	62,
										25,	34,	57,	62,
											43,	67,	72,
												93,	99,
													105
]

[model.eval.renju]
model_type = 2
table2 = [
0,	0,	0,	0,	0,	1,	1,	1,	2,	9,	9,	15,	36,	40,
	0,	0,	0,	0,	1,	1,	1,	2,	9,	9,	15,	36,	40,
		0,	0,	0,	1,	1,	1,	2,	9,	9,	15,	36,	40,
			0,	0,	1,	1,	1,	2,	9,	9,	15,	37,	40,
				1,	2,	2,	2,	3,	10,	10,	17,	38,	42,
					3,	3,	3,	4,	12,	12,	18,	40,	44,
						3,	3,	4,	12,	12,	18,	40,	44,
							3,	4,	12,	12,	18,	40,	44,
								5,	13,	13,	20,	42,	46,
									23,	23,	30,	54,	58,
										23,	30,	54,	58,
											38,	63,	67,
												90,	94,
													99
]

[model.score]
[model.score.self]
model_type = 1
table1 = [-1,0,0,1,2,6,6,6,9,16,20,30,100,500]
table1_scale = 2.0

[model.score.oppo]
model_type = 1
table1 = [-1,0,0,1,2,6,6,6,9,16,20,30,100,500]
table1_scale = 1.0
)internalConfig";
