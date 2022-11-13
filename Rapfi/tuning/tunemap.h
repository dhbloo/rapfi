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

#include "../core/iohelper.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>

namespace Tuning {

/// The default parameter range init function.
template <typename T>
constexpr auto defaultRange(T v) -> std::pair<T, T>
{
    return v > T(0) ? std::pair<T, T>(T(0), 2 * v) : std::pair<T, T>(2 * v, T(0));
}

/// A hyper-parameter tuning facility which records the mapping of a parameter name
/// to its value reference, which are updated from the input options.
class TuneMap
{
    using PostUpdateHook            = std::function<void()>;
    TuneMap()                       = default;
    TuneMap(const TuneMap &)        = delete;
    void operator=(const TuneMap &) = delete;

    /// Singleton instance
    static TuneMap &instance()
    {
        static TuneMap t;
        return t;
    }

    struct EntryBase
    {
        virtual ~EntryBase()                = default;
        virtual void init()                 = 0;
        virtual bool read(std::istream &is) = 0;
    };

    template <typename T>
    struct Entry : public EntryBase
    {
        using Range        = std::pair<T, T>;
        using RangeInitFun = std::function<Range(T)>;

        Entry(const std::string &n, T &v, Range r, PostUpdateHook post)
            : name(n)
            , value(v)
            , range(r)
            , rangeInit(nullptr)
            , postUpdate(post)
        {}
        Entry(const std::string &n, T &v, RangeInitFun init, PostUpdateHook post)
            : name(n)
            , value(v)
            , range()
            , rangeInit(init)
            , postUpdate(post)
        {}
        void operator=(const Entry &) = delete;
        void init() override
        {
            if (rangeInit)
                range = rangeInit(value);

            // Print formatted parameters, ready to be copy-pasted for external tuner
            MESSAGEL(name << "," << value << "," << range.first << "," << range.second << ","
                          << ((double)range.second - (double)range.first) / 20.0 << ","
                          << "0.0020");
        }
        bool read(std::istream &is) override
        {
            if constexpr (std::is_enum_v<T>) {
                using ValType = std::underlying_type_t<T>;
                ValType newValue;
                if (is >> newValue) {
                    newValue = std::clamp<ValType>(newValue,
                                                   static_cast<ValType>(range.first),
                                                   static_cast<ValType>(range.second));
                    value    = static_cast<T>(newValue);
                    if (postUpdate)
                        postUpdate();
                    return true;
                }
            }
            else {
                T newValue;
                if (is >> newValue) {
                    if (std::is_integral_v<T> && is.peek() == '.') {
                        double fraction;
                        // Round to next int if the fraction part is larger than 0.5
                        if (is >> fraction && std::round(fraction) > 0)
                            newValue += 1;
                    }

                    value = std::clamp<T>(newValue, range.first, range.second);
                    if (postUpdate)
                        postUpdate();
                    return true;
                }
            }
            return false;
        }

        std::string    name;
        T             &value;
        Range          range;
        RangeInitFun   rangeInit;
        PostUpdateHook postUpdate;
    };

    template <typename T>
    void add(const std::string &name, T &value, T min, T max, PostUpdateHook postUpdate = nullptr)
    {
        map[name] = std::make_unique<Entry<T>>(name, value, std::make_pair(min, max), postUpdate);
    }

    template <typename T>
    void add(const std::string              &name,
             T                              &value,
             typename Entry<T>::RangeInitFun rangeInit,
             PostUpdateHook                  postUpdate = nullptr)
    {
        map[name] = std::make_unique<Entry<T>>(name, value, rangeInit, postUpdate);
    }

    template <typename T>
    void add(const std::string &name, T &value, PostUpdateHook postUpdate = nullptr)
    {
        map[name] = std::make_unique<Entry<T>>(name, value, defaultRange<T>, postUpdate);
    }

    // Template specialization for arrays: recursively handle multi-dimensional arrays
    template <typename T, size_t N, typename... Args>
    void add(const std::string &name, T (&value)[N], Args &&...args)
    {
        for (size_t i = 0; i < N; i++)
            add(name + "[" + std::to_string(i) + "]", value[i], args...);
    }

    std::map<std::string, std::unique_ptr<EntryBase>> map;

public:
    template <typename... Args>
    static int addNewParam(Args &&...args)
    {
        instance().add(std::forward<Args>(args)...);
        return 0;
    }
    /// Init all added hyper-parameter ranges and print out tuning infos.
    static void init()
    {
        for (auto &e : instance().map)
            e.second->init();
    }
    /// Checks if current option name is a hyper-parameter, if so, read a value
    /// from the given input stream and return true.
    static bool tryReadOption(const std::string &name, std::istream &is)
    {
        auto it = instance().map.find(name);
        if (it != instance().map.end()) {
            it->second->read(is);
            return true;
        }
        return false;
    }
};

#define STRINGIFY2(x)     #x
#define STRINGIFY(x)      STRINGIFY2(x)
#define UNIQUE2(x, y)     x##y
#define UNIQUE(x, y)      UNIQUE2(x, y)  // Two indirection levels to expand __LINE__
#define FIRST_ARG(x, ...) x

/// Our tune macro for adding new parameters to tunemap which can be controlled from protocol.
/// Usage:
///     1. Tune single variable or an array of variables of basic type or enum:
///         int x = 10, y[4] = {10, 20, 30, 40};
///         Value v = Value(200);
///         TUNE(x); TUNE(y); TUNE(v);
///     2. Tune variable with fixed range:
///         float x = 1.0f;
///         TUNE(x, 0.0f, 10.0f);
///     3. Tune variable with dynamic range depending on the initial parameter value:
///         float x = 1.0f;
///         TUNE(x, [](float v) { return std::make_pair(v - 10, v + 10); });
///     4. Call post update function after a new value is read:
///         float x = 1.0f;
///         TUNE(x, []() { std::cout << "x is updated to " << x; });
#define TUNE(...)             \
    int UNIQUE(p, __LINE__) = \
        ::Tuning::TuneMap::addNewParam(STRINGIFY(FIRST_ARG(__VA_ARGS__)), __VA_ARGS__)

}  // namespace Tuning
